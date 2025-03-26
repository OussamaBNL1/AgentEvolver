// This file exists for backward compatibility
// It doesn't actually connect to MongoDB directly anymore
// Instead, it provides compatibility with the backend API

import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

// Create a mock connection function that logs a message
export async function connectToDatabase() {
  console.log('Using backend API instead of direct MongoDB connection');
  return {
    db: null,
    mongoose: null
  };
}

// Create a mock function to check connection status
export function getConnectionStatus() {
  return {
    isConnected: true,
    readyState: 1,
    host: API_URL,
    name: 'workit-api'
  };
}

// Create a mock function to disconnect
export async function disconnectFromDatabase() {
  console.log('No direct MongoDB connection to disconnect');
}

// Export a dummy mongoose object for compatibility
export const mongoose = {
  connection: {
    readyState: 1,
    db: null,
    host: API_URL,
    name: 'workit-api'
  },
  model: () => {
    console.warn('Using mongoose.model is deprecated. Please use the API instead.');
    return {};
  },
  Schema: class Schema {
    constructor() {
      console.warn('Using mongoose.Schema is deprecated. Please use the API instead.');
    }
  }
};
