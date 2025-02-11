from fastapi import FastAPI, File, UploadFile
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = FastAPI()

# Database setup
DATABASE_URL = "postgresql://myuser:mypassword@db:5432/mydatabase"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define a model
class MediaAsset(Base):
    __tablename__ = 'media_assets'
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content_type = Column(String)

# Create the database tables
Base.metadata.create_all(bind=engine)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Save file details to the database
    db = SessionLocal()
    media_asset = MediaAsset(filename=file.filename, content_type=file.content_type)
    db.add(media_asset)
    db.commit()
    db.refresh(media_asset)
    db.close()
    
    return {"filename": file.filename, "content_type": file.content_type} 