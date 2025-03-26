import mongoose from 'mongoose';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Get the directory name
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables from the root .env file
dotenv.config({ path: path.join(__dirname, '../../.env') });

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/workit';

console.log(`Attempting to connect to MongoDB at: ${MONGODB_URI}`);

// Try to connect to MongoDB
mongoose.connect(MONGODB_URI)
  .then(() => {
    console.log('✅ Successfully connected to MongoDB!');
    console.log('Connection details:');
    console.log(`  - Host: ${mongoose.connection.host}`);
    console.log(`  - Database: ${mongoose.connection.name}`);
    console.log(`  - Ready state: ${mongoose.connection.readyState} (1 = connected)`);
    console.log('\nYour MongoDB Compass connection string should be:');
    console.log(MONGODB_URI);
    console.log('\nYou can paste this into MongoDB Compass to connect.');
  })
  .catch(err => {
    console.error('❌ Failed to connect to MongoDB:');
    console.error(err);
    console.log('\nPossible reasons for connection failure:');
    console.log('  - MongoDB is not running');
    console.log('  - Connection string is incorrect');
    console.log('  - Network issues or firewall blocking the connection');
    console.log('  - Authentication issues (if using authentication)');
    console.log('\nSuggested actions:');
    console.log('  - Verify MongoDB is installed and running');
    console.log('  - Check the connection string in your .env file');
    console.log('  - Try connecting through MongoDB Compass manually');
    console.log('  - Ensure network/firewall allows MongoDB connections');
  })
  .finally(() => {
    // Close the connection after 2 seconds
    setTimeout(() => {
      mongoose.connection.close();
      console.log('\nConnection closed.');
      process.exit(0);
    }, 2000);
  });
