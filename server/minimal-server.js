import express from 'express';
import cors from 'cors';

const app = express();
const PORT = 5001;

// Middleware
app.use(cors());
app.use(express.json());

// Test route
app.get('/api', (req, res) => {
  res.json({ message: 'WorkiT API is running (minimal server)' });
});

// Services routes
app.get('/api/services', (req, res) => {
  // Mock service data
  const services = [
    {
      _id: '1',
      title: 'Développement d\'application web React',
      description: 'Création d\'applications web modernes et réactives avec React et Next.js',
      price: 200,
      category: 'web',
      subcategory: 'frontend',
      features: ['Interface responsive', 'API RESTful', 'Authentification'],
      deliveryTime: 10,
      revisions: '3',
      userId: '1',
      image: 'https://images.unsplash.com/photo-1593720213428-28a5b9e94613?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8d2ViJTIwZGV2ZWxvcG1lbnR8ZW58MHx8MHx8fDA%3D'
    },
    {
      _id: '2',
      title: 'Création d\'applications mobiles iOS et Android',
      description: 'Développement d\'applications mobiles cross-platform avec React Native',
      price: 300,
      category: 'mobile',
      subcategory: 'react-native',
      features: ['UI native', 'Notifications push', 'Offline mode'],
      deliveryTime: 15,
      revisions: '2',
      userId: '1',
      image: 'https://images.unsplash.com/photo-1581291518633-83b4ebd1d83e?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fG1vYmlsZSUyMGFwcHxlbnwwfHwwfHx8MA%3D%3D'
    },
    {
      _id: '3',
      title: 'Rédaction SEO et copywriting',
      description: 'Création de contenu optimisé pour les moteurs de recherche et engageant pour vos utilisateurs',
      price: 80,
      category: 'writing',
      subcategory: 'seo',
      features: ['Recherche de mots-clés', 'Optimisation SEO', 'Style captivant'],
      deliveryTime: 3,
      revisions: '5',
      userId: '2',
      image: 'https://images.unsplash.com/photo-1499750310107-5fef28a66643?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTN8fHdyaXRpbmd8ZW58MHx8MHx8fDA%3D'
    }
  ];

  res.json({
    services,
    totalPages: 1,
    currentPage: 1,
    totalServices: services.length
  });
});

// Get service by ID
app.get('/api/services/:id', (req, res) => {
  const id = req.params.id;

  // Mock service data
  const services = [
    {
      _id: '1',
      title: 'Développement d\'application web React',
      description: 'Création d\'applications web modernes et réactives avec React et Next.js',
      price: 200,
      category: 'web',
      subcategory: 'frontend',
      features: ['Interface responsive', 'API RESTful', 'Authentification'],
      deliveryTime: 10,
      revisions: '3',
      userId: '1',
      image: 'https://images.unsplash.com/photo-1593720213428-28a5b9e94613?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8d2ViJTIwZGV2ZWxvcG1lbnR8ZW58MHx8MHx8fDA%3D'
    },
    {
      _id: '2',
      title: 'Création d\'applications mobiles iOS et Android',
      description: 'Développement d\'applications mobiles cross-platform avec React Native',
      price: 300,
      category: 'mobile',
      subcategory: 'react-native',
      features: ['UI native', 'Notifications push', 'Offline mode'],
      deliveryTime: 15,
      revisions: '2',
      userId: '1',
      image: 'https://images.unsplash.com/photo-1581291518633-83b4ebd1d83e?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fG1vYmlsZSUyMGFwcHxlbnwwfHwwfHx8MA%3D%3D'
    },
    {
      _id: '3',
      title: 'Rédaction SEO et copywriting',
      description: 'Création de contenu optimisé pour les moteurs de recherche et engageant pour vos utilisateurs',
      price: 80,
      category: 'writing',
      subcategory: 'seo',
      features: ['Recherche de mots-clés', 'Optimisation SEO', 'Style captivant'],
      deliveryTime: 3,
      revisions: '5',
      userId: '2',
      image: 'https://images.unsplash.com/photo-1499750310107-5fef28a66643?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTN8fHdyaXRpbmd8ZW58MHx8MHx8fDA%3D'
    }
  ];

  const service = services.find(s => s._id === id);

  if (!service) {
    return res.status(404).json({ message: 'Service not found' });
  }

  res.json(service);
});

// Users routes - basic implementation
app.get('/api/users', (req, res) => {
  const users = [
    {
      _id: '1',
      username: 'thomas',
      fullName: 'Thomas Bernard',
      isFreelancer: true,
      profilePicture: 'https://randomuser.me/api/portraits/men/32.jpg'
    },
    {
      _id: '2',
      username: 'julie',
      fullName: 'Julie Moreau',
      isFreelancer: true,
      profilePicture: 'https://randomuser.me/api/portraits/women/44.jpg'
    }
  ];

  res.json(users);
});

// Login route
app.post('/api/users/login', (req, res) => {
  const { email, password } = req.body;

  // Simple mock login (in a real app, this would verify credentials properly)
  if (email && password) {
    res.json({
      user: {
        _id: '1',
        username: 'thomas',
        email: email,
        fullName: 'Thomas Bernard',
        isFreelancer: true,
        profilePicture: 'https://randomuser.me/api/portraits/men/32.jpg'
      },
      token: 'mock-jwt-token'
    });
  } else {
    res.status(401).json({ message: 'Invalid credentials' });
  }
});

// Connection status endpoint
app.get('/api/dbstatus', (req, res) => {
  res.json({
    isConnected: false,
    readyState: 0,
    host: 'none',
    name: 'none',
    useMockData: true
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Minimal server running on port ${PORT}`);
});
