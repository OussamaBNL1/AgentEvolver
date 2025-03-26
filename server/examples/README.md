# MongoDB Sample Data

This directory contains sample data that you can import into MongoDB to get started with the WorkiT platform.

## How to Import the Sample Data into MongoDB Compass

1. **Start MongoDB Compass** and connect to your database (either local or cloud-based)

2. **Create a new database** called `workit` if it doesn't already exist

3. **For each collection import:**
   - In MongoDB Compass, navigate to the `workit` database
   - Click on "Create Collection" if the collection doesn't exist (users, services, jobs, etc.)
   - In each collection view, click on "Add Data" and select "Import File"
   - Select the `mongodb-sample-data.json` file from this directory
   - Choose "JSON" as the file format
   - When importing, use the collection name from the JSON file

4. **Alternative import method using mongoimport tool:**
   If you have the MongoDB Database Tools installed, you can use the `mongoimport` command:

   ```bash
   # For users collection
   mongoimport --db workit --collection users --file mongodb-sample-data.json --jsonArray

   # For services collection
   mongoimport --db workit --collection services --file mongodb-sample-data.json --jsonArray

   # For jobs collection
   mongoimport --db workit --collection jobs --file mongodb-sample-data.json --jsonArray
   ```

## Data Structure

The sample data includes:

- **Users**: Freelancers and clients with their profiles
- **Services**: Services offered by freelancers
- **Jobs**: Job opportunities posted by clients

## Connection String

To connect to your local MongoDB instance:

```
mongodb://localhost:27017/workit
```

For MongoDB Atlas, use the connection string provided in your Atlas dashboard.

## Notes

- The sample data includes placeholder passwords ('password123'). In a production environment, always use securely hashed passwords.
- Object IDs are pre-generated to maintain proper references between collections.
- Make sure MongoDB is running before attempting to import the data.
