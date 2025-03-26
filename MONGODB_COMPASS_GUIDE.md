# MongoDB Compass Connection Guide for WorkiT

This guide will help you connect MongoDB Compass to the WorkiT platform's backend database.

## Prerequisites

1. [MongoDB Compass](https://www.mongodb.com/try/download/compass) installed on your machine
2. MongoDB server running (either locally or in the cloud)

## Connection Steps

### Option 1: Connect to a Local MongoDB Instance

1. Make sure MongoDB is installed and running on your local machine
2. Open MongoDB Compass
3. Use the following connection string:
   ```
   mongodb://localhost:27017/workit
   ```
4. Click "Connect"

### Option 2: Connect to a Cloud MongoDB Instance (MongoDB Atlas)

1. Create a MongoDB Atlas account if you don't have one
2. Set up a cluster and get your connection string
3. Open MongoDB Compass
4. Paste your MongoDB Atlas connection string (it will look something like):
   ```
   mongodb+srv://<username>:<password>@cluster0.mongodb.net/workit
   ```
5. Replace `<username>` and `<password>` with your actual credentials
6. Click "Connect"

## Database Structure

The WorkiT database contains the following collections:

1. **users** - User profiles including freelancers and clients
2. **services** - Services offered by freelancers
3. **jobs** - Job listings posted by clients
4. **messages** - Messages exchanged between users
5. **conversations** - Conversation threads between users
6. **orders** - Service orders and transactions

## Working with the Collections

### Users Collection

Each document in the users collection has the following structure:

```json
{
  "_id": ObjectId("..."),
  "username": "thomas",
  "email": "thomas@workit.com",
  "password": "hashed_password",
  "fullName": "Thomas Bernard",
  "isFreelancer": true,
  "profilePicture": "https://example.com/profile.jpg",
  "bio": "Full-stack developer with 5 years of experience",
  "skills": ["React", "Node.js", "MongoDB"],
  "location": "Tunis",
  "contactInfo": "thomas@workit.com",
  "joinedDate": ISODate("2021-05-10T00:00:00Z"),
  "isAdmin": false,
  "createdAt": ISODate("2021-05-10T00:00:00Z"),
  "updatedAt": ISODate("2021-05-10T00:00:00Z")
}
```

### Services Collection

Each document in the services collection has the following structure:

```json
{
  "_id": ObjectId("..."),
  "title": "Web Development with React",
  "category": "web",
  "subcategory": "frontend",
  "price": 200,
  "description": "I will create responsive web applications using React",
  "features": ["Responsive Design", "API Integration", "Authentication"],
  "deliveryTime": 10,
  "revisions": "3",
  "userId": ObjectId("..."),
  "image": "https://example.com/service.jpg",
  "gallery": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
  "createdAt": ISODate("2021-06-15T00:00:00Z"),
  "updatedAt": ISODate("2021-06-15T00:00:00Z")
}
```

## Troubleshooting

If you encounter connection issues:

1. Verify MongoDB is running
2. Check firewall settings
3. Ensure your connection string is correct
4. Verify network connectivity
5. Check if authentication credentials are correct

For more help, visit the [MongoDB Compass documentation](https://docs.mongodb.com/compass/current/).

## Development Environment

For local development, the WorkiT platform uses the following environment variables:

```
MONGODB_URI=mongodb://localhost:27017/workit
PORT=5001
NODE_ENV=development
```

You can modify these in the `.env` file in the project root.
