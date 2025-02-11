from fastapi import FastAPI, File, UploadFile, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path
from datetime import datetime
import magic
from PIL import Image
from moviepy.editor import VideoFileClip
from mutagen import File as MutagenFile
import platform
from stat import ST_MTIME, ST_CTIME
import tempfile
import shutil
from PIL.PngImagePlugin import PngImageFile
import subprocess
import json
import re

app = FastAPI()

# Database setup
DATABASE_URL = "postgresql://myuser:mypassword@db:5432/mydatabase"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define base upload directory
UPLOAD_DIR = Path("uploads")

# Ensure upload directories exist
for dir_name in ["images", "videos", "documents", "audio", "other"]:
    (UPLOAD_DIR / dir_name).mkdir(parents=True, exist_ok=True)

# Define content type mappings
CONTENT_TYPE_MAPPING = {
    "images": ["image/jpeg", "image/png", "image/gif", "image/webp"],
    "videos": ["video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo"],
    "audio": ["audio/mpeg", "audio/wav", "audio/ogg", "audio/x-m4a"],
    "documents": ["application/pdf", "application/msword", 
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain"]
}

class MediaAsset(Base):
    __tablename__ = 'media_assets'
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content_type = Column(String)
    file_path = Column(String)
    
    # Common metadata
    file_size = Column(Integer)  # in bytes
    file_created_date = Column(DateTime, nullable=True)
    file_modified_date = Column(DateTime, nullable=True)
    original_creation_date = Column(DateTime, nullable=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    
    # Image specific
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    
    # Video/Audio specific
    duration = Column(Float, nullable=True)  # in seconds
    bitrate = Column(Integer, nullable=True)  # in bits per second

# Create the database tables
Base.metadata.create_all(bind=engine)

def get_file_category(content_type: str) -> str:
    """Determine the appropriate category for a file based on its content type."""
    for category, types in CONTENT_TYPE_MAPPING.items():
        if content_type in types:
            return category
    return "other"

def get_file_dates(file_path: Path) -> dict:
    """Get various timestamps for a file."""
    stats = file_path.stat()
    dates = {
        'file_modified_date': datetime.fromtimestamp(stats.st_mtime),
        'file_created_date': None,
        'original_creation_date': None
    }
    
    try:
        if platform.system() == 'Windows':
            # On Windows, st_ctime is creation time
            dates['file_created_date'] = datetime.fromtimestamp(stats.st_ctime)
        else:
            # On Unix, try to get birth time (not always available)
            try:
                # Try to get birth time (macOS)
                dates['file_created_date'] = datetime.fromtimestamp(stats.st_birthtime)
            except AttributeError:
                # Fallback to ctime on other Unix systems
                dates['file_created_date'] = datetime.fromtimestamp(stats.st_ctime)
        
        # Try to get metadata creation date from file properties
        # This might be available in EXIF data for images, or other metadata
        if hasattr(stats, 'st_birthtime'):
            dates['original_creation_date'] = datetime.fromtimestamp(stats.st_birthtime)
    except Exception as e:
        print(f"Error getting file dates: {e}")
    
    return dates

def extract_ffmpeg_metadata(file_path: Path) -> dict:
    """Extract metadata from a file using ffmpeg."""
    metadata = {}
    try:
        # Run ffprobe (part of ffmpeg) to get metadata in JSON format
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(file_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Extract dates from metadata
            format_tags = data.get('format', {}).get('tags', {})
            
            # Try different date fields
            date_fields = [
                'creation_time',
                'date',
                'date_created',
                'DateTimeOriginal',
                'modify_date',
                'encoded_date'
            ]
            
            for field in date_fields:
                if field in format_tags:
                    try:
                        date_str = format_tags[field]
                        # Handle different date formats
                        date_formats = [
                            '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO format with microseconds
                            '%Y-%m-%d %H:%M:%S',      # Standard format
                            '%Y:%m:%d %H:%M:%S',      # EXIF format
                            '%Y-%m-%d',               # Simple date
                            '%Y:%m:%d'                # Alternative date format
                        ]
                        
                        for fmt in date_formats:
                            try:
                                parsed_date = datetime.strptime(date_str, fmt)
                                if 'creation' in field or 'created' in field or 'Original' in field:
                                    metadata['original_creation_date'] = parsed_date
                                elif 'modify' in field:
                                    metadata['file_modified_date'] = parsed_date
                                break
                            except ValueError:
                                continue
                    except Exception as e:
                        print(f"Error parsing date from {field}: {e}")
            
            # Extract basic metadata
            if 'duration' in data['format']:
                metadata['duration'] = float(data['format']['duration'])
            if 'bit_rate' in data['format']:
                metadata['bitrate'] = int(data['format']['bit_rate'])
            
            # Extract dimensions from video/image streams
            for stream in data.get('streams', []):
                if stream['codec_type'] in ['video', 'image']:
                    metadata['width'] = stream.get('width')
                    metadata['height'] = stream.get('height')
                    break
            
    except Exception as e:
        print(f"Error extracting ffmpeg metadata: {e}")
    
    return metadata

async def extract_original_metadata(file: UploadFile, content: bytes) -> dict:
    """Extract metadata from the uploaded file before saving."""
    current_time = datetime.utcnow()
    metadata = {
        'file_size': len(content),
        'file_created_date': None,
        'file_modified_date': None,
        'original_creation_date': None,
        'upload_date': current_time
    }

    # Create a temporary file to analyze
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)
    
    try:
        # Try to extract metadata using ffmpeg first
        ffmpeg_metadata = extract_ffmpeg_metadata(temp_path)
        metadata.update(ffmpeg_metadata)
        
        # If ffmpeg didn't get all the metadata we want, try other methods
        if not metadata.get('original_creation_date'):
            # Image metadata using PIL
            if file.content_type.startswith('image/'):
                try:
                    with Image.open(temp_path) as img:
                        metadata.update({
                            'width': metadata.get('width', img.width),
                            'height': metadata.get('height', img.height),
                        })
                        
                        # Try to get original creation date from EXIF data
                        if hasattr(img, '_getexif') and img._getexif():
                            exif = img._getexif()
                            print(f"EXIF data: {exif}")  # Debugging line
                            date_tags = [36867, 36868, 306, 50971, 31052]
                            
                            for tag in date_tags:
                                if exif and tag in exif and exif[tag]:
                                    try:
                                        date_str = str(exif[tag]).strip()
                                        metadata['original_creation_date'] = datetime.strptime(
                                            date_str, 
                                            '%Y:%m:%d %H:%M:%S'
                                        )
                                        break
                                    except Exception as e:
                                        print(f"Error parsing EXIF date from tag {tag}: {e}")
                except Exception as e:
                    print(f"Error extracting image metadata: {e}")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            print(f"Error cleaning up temp file: {e}")
    
    return metadata

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    title: str = None,
    description: str = None
):
    try:
        # Read file content
        content = await file.read()
        
        # Extract metadata before saving
        metadata = await extract_original_metadata(file, content)
        
        # Determine the appropriate directory based on content type
        category = get_file_category(file.content_type)
        
        # Create a unique filename to avoid conflicts
        file_location = UPLOAD_DIR / category / file.filename
        counter = 1
        while file_location.exists():
            stem = Path(file.filename).stem
            suffix = Path(file.filename).suffix
            new_filename = f"{stem}_{counter}{suffix}"
            file_location = UPLOAD_DIR / category / new_filename
            counter += 1

        # Save the file
        with open(file_location, "wb") as buffer:
            buffer.write(content)
        
        # Save file details to the database
        db = SessionLocal()
        media_asset = MediaAsset(
            filename=file_location.name,
            content_type=file.content_type,
            file_path=str(file_location),
            title=title or file_location.stem,
            description=description,
            file_size=metadata['file_size'],
            file_created_date=metadata['file_created_date'],
            file_modified_date=metadata['file_modified_date'],
            original_creation_date=metadata['original_creation_date'],
            width=metadata.get('width'),
            height=metadata.get('height'),
            duration=metadata.get('duration'),
            bitrate=metadata.get('bitrate')
        )
        db.add(media_asset)
        db.commit()
        db.refresh(media_asset)
        db.close()
        
        return {
            "filename": file_location.name,
            "content_type": file.content_type,
            "category": category,
            "file_path": str(file_location),
            "metadata": metadata
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 