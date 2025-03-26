import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import mongoose from 'mongoose';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import serviceRoutes from './routes/serviceRoutes.js';
import userRoutes from './routes/userRoutes.js';
import messageRoutes from './routes/messageRoutes.js';
import jobRoutes from './routes/jobRoutes.js';

// Configuration
dotenv.config();
const app = express();
// Use environment PORT variable or fallback to 5001
const PORT = process.env.PORT || 5001;
console.log(`Using PORT: ${PORT}`);
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// MongoDB Connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/workit';
let isMongoConnected = false;

// Try to connect to MongoDB, but continue with mock data if it fails
mongoose.connect(MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => {
    console.log(`✅ Connected to MongoDB: ${mongoose.connection.host}`);
    isMongoConnected = true;
  })
  .catch(err => {
    console.error('❌ MongoDB connection error:', err.message);
    console.log('⚠️ Continuing with mock data for development...');
  });


// Mock data middleware for development when MongoDB is not available
app.use((req, res, next) => {
  // Only use mock data if MongoDB is not connected
  if (!isMongoConnected && process.env.NODE_ENV === 'development') {
    req.useMockData = true;
  }
  next();
});

// Serve static files in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(join(__dirname, '../dist')));
}

// Routes
app.use('/api/services', serviceRoutes);
app.use('/api/users', userRoutes);
app.use('/api/messages', messageRoutes);
app.use('/api/jobs', jobRoutes);

// MongoDB Connection Status endpoint
app.get('/api/dbstatus', (req, res) => {
  res.json({
    isConnected: mongoose.connection.readyState === 1,
    readyState: mongoose.connection.readyState,
    host: mongoose.connection.host || 'none',
    name: mongoose.connection.name || 'none',
    useMockData: !isMongoConnected
  });
});

// Root route for testing
app.get('/api', (req, res) => {
  res.json({
    message: 'WorkiT API is running',
    mongoStatus: isMongoConnected ? 'connected' : 'using mock data'
  });
});

// Catch-all route for client-side routing in production
if (process.env.NODE_ENV === 'production') {
  app.get('*', (req, res) => {
    res.sendFile(join(__dirname, '../dist/index.html'));
  });
}

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

export default app;
