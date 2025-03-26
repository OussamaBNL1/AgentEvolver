import express from 'express';
import Service from '../models/Service.js';

const router = express.Router();

// Mock data for development when MongoDB is not available
const mockServices = [
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
    image: 'https://images.unsplash.com/photo-1593720213428-28a5b9e94613?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8d2ViJTIwZGV2ZWxvcG1lbnR8ZW58MHx8MHx8fDA%3D',
    createdAt: new Date('2022-01-15'),
    updatedAt: new Date('2022-01-15')
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
    image: 'https://images.unsplash.com/photo-1581291518633-83b4ebd1d83e?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fG1vYmlsZSUyMGFwcHxlbnwwfHwwfHx8MA%3D%3D',
    createdAt: new Date('2022-02-10'),
    updatedAt: new Date('2022-02-10')
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
    image: 'https://images.unsplash.com/photo-1499750310107-5fef28a66643?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTN8fHdyaXRpbmd8ZW58MHx8MHx8fDA%3D',
    createdAt: new Date('2022-03-05'),
    updatedAt: new Date('2022-03-05')
  }
];

// Get all services
router.get('/', async (req, res) => {
  try {
    const { category, search, sort, limit = 10, page = 1 } = req.query;
    const skip = (parseInt(page) - 1) * parseInt(limit);

    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      // Apply filters to mock data
      let filteredServices = [...mockServices];

      if (category) {
        filteredServices = filteredServices.filter(s => s.category === category);
      }

      if (search) {
        const searchLower = search.toLowerCase();
        filteredServices = filteredServices.filter(s =>
          s.title.toLowerCase().includes(searchLower) ||
          s.description.toLowerCase().includes(searchLower)
        );
      }

      // Apply sorting
      if (sort === 'price-asc') {
        filteredServices.sort((a, b) => a.price - b.price);
      } else if (sort === 'price-desc') {
        filteredServices.sort((a, b) => b.price - a.price);
      } else if (sort === 'newest') {
        filteredServices.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
      } else {
        // Default sort by newest
        filteredServices.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
      }

      // Apply pagination
      const paginatedServices = filteredServices.slice(skip, skip + parseInt(limit));

      return res.json({
        services: paginatedServices,
        totalPages: Math.ceil(filteredServices.length / parseInt(limit)),
        currentPage: parseInt(page),
        totalServices: filteredServices.length
      });
    }

    // If MongoDB is available, use it
    const query = {};
    if (category) {
      query.category = category;
    }
    if (search) {
      query.$text = { $search: search };
    }

    // Build sort
    const sortOptions = {};
    if (sort === 'price-asc') {
      sortOptions.price = 1;
    } else if (sort === 'price-desc') {
      sortOptions.price = -1;
    } else if (sort === 'newest') {
      sortOptions.createdAt = -1;
    } else {
      // Default sort by newest
      sortOptions.createdAt = -1;
    }

    const services = await Service.find(query)
      .sort(sortOptions)
      .limit(parseInt(limit))
      .skip(skip)
      .exec();

    const total = await Service.countDocuments(query);

    res.json({
      services,
      totalPages: Math.ceil(total / parseInt(limit)),
      currentPage: parseInt(page),
      totalServices: total
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Get a specific service
router.get('/:id', async (req, res) => {
  try {
    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      const service = mockServices.find(s => s._id === req.params.id);
      if (!service) {
        return res.status(404).json({ message: 'Service not found' });
      }
      return res.json(service);
    }

    // If MongoDB is available, use it
    const service = await Service.findById(req.params.id);
    if (!service) {
      return res.status(404).json({ message: 'Service not found' });
    }
    res.json(service);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Create a new service
router.post('/', async (req, res) => {
  try {
    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      const newService = {
        _id: (mockServices.length + 1).toString(),
        ...req.body,
        createdAt: new Date(),
        updatedAt: new Date()
      };
      mockServices.push(newService);
      return res.status(201).json(newService);
    }

    // If MongoDB is available, use it
    const service = new Service(req.body);
    const savedService = await service.save();
    res.status(201).json(savedService);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Update a service
router.put('/:id', async (req, res) => {
  try {
    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      const serviceIndex = mockServices.findIndex(s => s._id === req.params.id);
      if (serviceIndex === -1) {
        return res.status(404).json({ message: 'Service not found' });
      }

      // Check if user is owner of service
      if (mockServices[serviceIndex].userId !== req.body.userId) {
        return res.status(403).json({ message: 'Not authorized to update this service' });
      }

      const updatedService = {
        ...mockServices[serviceIndex],
        ...req.body,
        updatedAt: new Date()
      };

      mockServices[serviceIndex] = updatedService;
      return res.json(updatedService);
    }

    // If MongoDB is available, use it
    const service = await Service.findById(req.params.id);
    if (!service) {
      return res.status(404).json({ message: 'Service not found' });
    }

    // Check if user is owner of service
    if (service.userId.toString() !== req.body.userId) {
      return res.status(403).json({ message: 'Not authorized to update this service' });
    }

    Object.assign(service, req.body);
    const updatedService = await service.save();
    res.json(updatedService);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Delete a service
router.delete('/:id', async (req, res) => {
  try {
    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      const serviceIndex = mockServices.findIndex(s => s._id === req.params.id);
      if (serviceIndex === -1) {
        return res.status(404).json({ message: 'Service not found' });
      }

      // Check if user is owner of service
      if (mockServices[serviceIndex].userId !== req.body.userId) {
        return res.status(403).json({ message: 'Not authorized to delete this service' });
      }

      mockServices.splice(serviceIndex, 1);
      return res.json({ message: 'Service deleted' });
    }

    // If MongoDB is available, use it
    const service = await Service.findById(req.params.id);
    if (!service) {
      return res.status(404).json({ message: 'Service not found' });
    }

    // Check if user is owner of service
    if (service.userId.toString() !== req.body.userId) {
      return res.status(403).json({ message: 'Not authorized to delete this service' });
    }

    await service.deleteOne();
    res.json({ message: 'Service deleted' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Get services by user ID
router.get('/user/:userId', async (req, res) => {
  try {
    // Use mock data if MongoDB is not available
    if (req.useMockData) {
      const userServices = mockServices.filter(s => s.userId === req.params.userId);
      return res.json(userServices);
    }

    // If MongoDB is available, use it
    const services = await Service.find({ userId: req.params.userId });
    res.json(services);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

export default router;
