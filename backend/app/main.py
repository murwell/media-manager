from fastapi import FastAPI, File, UploadFile, HTTPException
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path

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
    file_path = Column(String)  # Add this new column

# Create the database tables
Base.metadata.create_all(bind=engine)

def get_file_category(content_type: str) -> str:
    """Determine the appropriate category for a file based on its content type."""
    for category, types in CONTENT_TYPE_MAPPING.items():
        if content_type in types:
            return category
    return "other"

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
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
        
        # Save file details to the database
        db = SessionLocal()
        media_asset = MediaAsset(
            filename=file_location.name,
            content_type=file.content_type,
            file_path=str(file_location)
        )
        db.add(media_asset)
        db.commit()
        db.refresh(media_asset)
        db.close()
        
        return {
            "filename": file_location.name,
            "content_type": file.content_type,
            "category": category,
            "file_path": str(file_location)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 