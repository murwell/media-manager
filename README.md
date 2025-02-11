## Large Files

Large files (>10MB) should not be committed to this repository. Instead:
- Store model files in ~/.opencv_models/
- Store uploads in the uploads/ directory (gitignored)
- Use external storage for large assets
- Document download procedures for required large files

## Development Setup

### Pre-commit Hook

To prevent accidentally committing large files, set up the pre-commit hook:

```bash
# Create the pre-commit hook
cat > .git/hooks/pre-commit << 'EOL'
#!/bin/bash

# Maximum file size in bytes (10MB = 10485760)
max_size=10485760

# Check all files being committed
git diff --cached --name-only | while read file; do
    # Skip if file is deleted
    if [ -f "$file" ]; then
        # Get file size
        size=$(stat -f%z "$file")
        if [ "$size" -gt $max_size ]; then
            echo "Error: $file is larger than ${max_size} bytes"
            exit 1
        fi
    fi
done
EOL

# Make it executable
chmod +x .git/hooks/pre-commit
```

This hook will prevent files larger than 10MB from being committed.

### Docker Setup

```bash
# Build and start containers
docker-compose up --build -d

# Run database migrations
docker-compose exec web alembic upgrade head
``` 