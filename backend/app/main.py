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

def extract_metadata(file_path: Path, content_type: str) -> dict:
    """Extract metadata based on file type."""
    metadata = {
        'file_size': file_path.stat().st_size,
        'original_creation_date': None
    }
    
    # Try to get creation date (platform dependent)
    try:
        if platform.system() == 'Windows':
            metadata['original_creation_date'] = datetime.fromtimestamp(file_path.stat().st_ctime)
        else:  # Unix-based systems
            metadata['original_creation_date'] = datetime.fromtimestamp(file_path.stat().st_birthtime)
    except:
        pass  # Creation date not available
    
    # Image metadata
    if content_type.startswith('image/'):
        try:
            with Image.open(file_path) as img:
                metadata.update({
                    'width': img.width,
                    'height': img.height,
                })
        except Exception as e:
            print(f"Error extracting image metadata: {e}")
    
    # Video metadata
    elif content_type.startswith('video/'):
        try:
            with VideoFileClip(str(file_path)) as video:
                metadata.update({
                    'duration': video.duration,
                    'width': video.size[0],
                    'height': video.size[1],
                })
        except Exception as e:
            print(f"Error extracting video metadata: {e}")
    
    # Audio metadata
    elif content_type.startswith('audio/'):
        try:
            audio = MutagenFile(str(file_path))
            if audio is not None:
                metadata.update({
                    'duration': audio.info.length if hasattr(audio.info, 'length') else None,
                    'bitrate': audio.info.bitrate if hasattr(audio.info, 'bitrate') else None,
                })
        except Exception as e:
            print(f"Error extracting audio metadata: {e}")
    
    return metadata

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    title: str = None,
    description: str = None
):
    try:
        # Determine the appropriate directory based on content type
        category = get_file_category(file.content_type)
        
        # Create a unique filename to avoid conflicts
        file_location = UPLOAD_DIR / category / file.filename
        
        # Ensure unique filename
        counter = 1
        while file_location.exists():
            stem = Path(file.filename).stem
            suffix = Path(file.filename).suffix
            new_filename = f"{stem}_{counter}{suffix}"
            file_location = UPLOAD_DIR / category / new_filename
            counter += 1

        # Save the file
        with open(file_location, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract metadata
        metadata = extract_metadata(file_location, file.content_type)
        
        # Save file details to the database
        db = SessionLocal()
        media_asset = MediaAsset(
            filename=file_location.name,
            content_type=file.content_type,
            file_path=str(file_location),
            title=title or file_location.stem,  # Use filename if no title provided
            description=description,
            file_size=metadata['file_size'],
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
            "metadata": {
                "file_size": metadata['file_size'],
                "original_creation_date": metadata['original_creation_date'],
                "upload_date": datetime.utcnow(),
                "width": metadata.get('width'),
                "height": metadata.get('height'),
                "duration": metadata.get('duration'),
                "bitrate": metadata.get('bitrate')
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 