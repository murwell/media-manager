from sqlalchemy import create_engine, text

# Use the same database URL as in main.py
DATABASE_URL = "postgresql://myuser:mypassword@db:5432/mydatabase"
engine = create_engine(DATABASE_URL)

def run_migrations():
    with engine.connect() as connection:
        connection.execute(text("""
            ALTER TABLE media_assets ADD COLUMN IF NOT EXISTS file_path VARCHAR;
        """))
        connection.commit()

if __name__ == "__main__":
    run_migrations() 