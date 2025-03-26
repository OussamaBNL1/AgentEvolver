import express from 'express';
import User from '../models/User.js';

const router = express.Router();

// Mock data for development when MongoDB is not available
const mockUsers = [
  {
    _id: '1',
    username: 'thomas',
    email: 'thomas@workit.com',
    password: 'password123',
    fullName: 'Thomas Bernard',
    isFreelancer: true,
    profilePicture: 'https://randomuser.me/api/portraits/men/32.jpg',
    bio: 'Développeur full stack avec 5 ans d\'expérience',
    skills: ['React', 'Node.js', 'TypeScript', 'MongoDB'],
    location: 'Tunis, Tunisie',
    contactInfo: 'thomas@workit.com',
    joinedDate: new Date('2021-05-10'),
    isAdmin: false,
    createdAt: new Date('2021-05-10'),
    updatedAt: new Date('2021-05-10')
  },
  {
    _id: '2',
    username: 'julie',
    email: 'julie@workit.com',
    password: 'password123',
    fullName: 'Julie Moreau',
    isFreelancer: true,
    profilePicture: 'https://randomuser.me/api/portraits/women/44.jpg',
    bio: 'Rédactrice SEO et copywriter professionnelle',
    skills: ['SEO', 'Copywriting', 'Content Strategy', 'Blogging'],
    location: 'Sousse, Tunisie',
    contactInfo: 'julie@workit.com',
    joinedDate: new Date('2021-06-15'),
    isAdmin: false,
    createdAt: new Date('2021-06-15'),
    updatedAt: new Date('2021-06-15')
  },
  {
    _id: '3',
    username: 'client',
    email: 'client@workit.com',
    password: 'password123',
    fullName: 'Client Test',
    isFreelancer: false,
    profilePicture: 'https://randomuser.me/api/portraits/men/67.jpg',
    bio: 'Entrepreneur à la recherche de talents',
    skills: [],
    location: 'Sfax, Tunisie',
    contactInfo: 'client@workit.com',
    joinedDate: new Date('2021-07-20'),
    isAdmin: false,
    createdAt: new Date('2021-07-20'),
    updatedAt: new Date('2021-07-20')
  }
];

// Get all users (admin only)
router.get('/', async (req, res) => {
  try {
    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      // Don't return passwords
      const usersWithoutPasswords = mockUsers.map(user => {
        const { password, ...userWithoutPassword } = user;
        return userWithoutPassword;
      });

      return res.json(usersWithoutPasswords);
    }

    // In a real app, add authentication middleware to verify admin status
    const users = await User.find().select('-password');
    res.json(users);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Get user by ID
router.get('/:id', async (req, res) => {
  try {
    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      const user = mockUsers.find(u => u._id === req.params.id);

      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      // Don't return password
      const { password, ...userWithoutPassword } = user;
      return res.json(userWithoutPassword);
    }

    const user = await User.findById(req.params.id).select('-password');
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    res.json(user);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Register a new user
router.post('/register', async (req, res) => {
  try {
    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      // Check if user already exists
      const userExists = mockUsers.find(u =>
        u.email === req.body.email || u.username === req.body.username
      );

      if (userExists) {
        if (userExists.email === req.body.email) {
          return res.status(400).json({ message: 'Email already in use' });
        } else {
          return res.status(400).json({ message: 'Username already taken' });
        }
      }

      // Create new user
      const newUser = {
        _id: (mockUsers.length + 1).toString(),
        ...req.body,
        joinedDate: new Date(),
        createdAt: new Date(),
        updatedAt: new Date()
      };

      mockUsers.push(newUser);

      // Don't return password
      const { password, ...userWithoutPassword } = newUser;
      return res.status(201).json(userWithoutPassword);
    }

    // Check if user already exists
    const userExists = await User.findOne({
      $or: [
        { email: req.body.email },
        { username: req.body.username }
      ]
    });

    if (userExists) {
      if (userExists.email === req.body.email) {
        return res.status(400).json({ message: 'Email already in use' });
      } else {
        return res.status(400).json({ message: 'Username already taken' });
      }
    }

    // In a real app, hash the password before saving
    const user = new User(req.body);
    const savedUser = await user.save();

    // Don't return the password
    const userWithoutPassword = savedUser.toObject();
    delete userWithoutPassword.password;

    res.status(201).json(userWithoutPassword);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Login user
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      // Find user by email
      const user = mockUsers.find(u => u.email === email);

      if (!user) {
        return res.status(401).json({ message: 'Invalid email or password' });
      }

      // Check password
      if (user.password !== password) {
        return res.status(401).json({ message: 'Invalid email or password' });
      }

      // Don't return password
      const { password: userPassword, ...userWithoutPassword } = user;

      return res.json({
        user: userWithoutPassword,
        token: 'mock-jwt-token'
      });
    }

    // Find user by email
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ message: 'Invalid email or password' });
    }

    // In a real app, compare hashed passwords
    if (user.password !== password) {
      return res.status(401).json({ message: 'Invalid email or password' });
    }

    // Don't return the password
    const userWithoutPassword = user.toObject();
    delete userWithoutPassword.password;

    // In a real app, generate JWT token here
    res.json({
      user: userWithoutPassword,
      token: 'mock-jwt-token'
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Update user
router.put('/:id', async (req, res) => {
  try {
    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      const userIndex = mockUsers.findIndex(u => u._id === req.params.id);

      if (userIndex === -1) {
        return res.status(404).json({ message: 'User not found' });
      }

      // Don't allow password updates through this route
      const updates = { ...req.body };
      delete updates.password;

      const updatedUser = {
        ...mockUsers[userIndex],
        ...updates,
        updatedAt: new Date()
      };

      mockUsers[userIndex] = updatedUser;

      // Don't return password
      const { password, ...userWithoutPassword } = updatedUser;
      return res.json(userWithoutPassword);
    }

    // In a real app, verify authorization (user can only update their own profile)
    const updates = { ...req.body };
    delete updates.password; // Don't allow password updates through this route

    const user = await User.findByIdAndUpdate(
      req.params.id,
      updates,
      { new: true, runValidators: true }
    ).select('-password');

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    res.json(user);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Change password
router.put('/:id/password', async (req, res) => {
  try {
    const { currentPassword, newPassword } = req.body;

    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      const userIndex = mockUsers.findIndex(u => u._id === req.params.id);

      if (userIndex === -1) {
        return res.status(404).json({ message: 'User not found' });
      }

      // Check current password
      if (mockUsers[userIndex].password !== currentPassword) {
        return res.status(401).json({ message: 'Current password is incorrect' });
      }

      // Update password
      mockUsers[userIndex].password = newPassword;
      mockUsers[userIndex].updatedAt = new Date();

      return res.json({ message: 'Password updated successfully' });
    }

    // Find user
    const user = await User.findById(req.params.id);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // In a real app, compare hashed passwords
    if (user.password !== currentPassword) {
      return res.status(401).json({ message: 'Current password is incorrect' });
    }

    // In a real app, hash the new password
    user.password = newPassword;
    await user.save();

    res.json({ message: 'Password updated successfully' });
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Get freelancers (users with isFreelancer = true)
router.get('/freelancers/all', async (req, res) => {
  try {
    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      const freelancers = mockUsers
        .filter(u => u.isFreelancer)
        .map(user => {
          const { password, ...userWithoutPassword } = user;
          return userWithoutPassword;
        });

      return res.json(freelancers);
    }

    const freelancers = await User.find({ isFreelancer: true }).select('-password');
    res.json(freelancers);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

export default router;
