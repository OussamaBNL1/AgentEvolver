import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

interface Service {
  id: string;
  title: string;
  category: string;
  subcategory: string;
  price: number;
  image: string;
  provider: {
    id: string;
    name: string;
    avatar?: string;
  };
  rating?: number;
  reviewCount?: number;
}

interface Category {
  id: string;
  name: string;
  subcategories: string[];
}

const ServicesPage = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedSubcategory, setSelectedSubcategory] = useState('all');
  const [priceRange, setPriceRange] = useState<[number, number]>([0, 1000]);
  const [services, setServices] = useState<Service[]>([]);
  const [displayedServices, setDisplayedServices] = useState<Service[]>([]);

  // Mock categories data
  const categories: Category[] = [
    {
      id: 'dev',
      name: 'Développement Web',
      subcategories: ['Frontend', 'Backend', 'Fullstack', 'WordPress', 'E-commerce'],
    },
    {
      id: 'design',
      name: 'Graphisme & Design',
      subcategories: ['Logo & Branding', 'Web Design', 'UI/UX Design', 'Illustrations', 'Bannières & Médias Sociaux'],
    },
    {
      id: 'marketing',
      name: 'Marketing Digital',
      subcategories: ['SEO', 'Médias Sociaux', 'Publicité', 'Email Marketing', 'Contenu'],
    },
    {
      id: 'video',
      name: 'Vidéo & Animation',
      subcategories: ['Montage Vidéo', 'Animation 2D/3D', 'Motion Graphics', 'Explainer Videos', 'Intros & Outros'],
    },
    {
      id: 'writing',
      name: 'Rédaction & Traduction',
      subcategories: ['Rédaction Web', 'Copywriting', 'Traduction', 'Relecture', 'Transcription'],
    },
  ];

  // Available subcategories based on selected category
  const availableSubcategories = selectedCategory === 'all'
    ? []
    : categories.find(cat => cat.id === selectedCategory)?.subcategories || [];

  useEffect(() => {
    // Mock services data
    const mockServices: Service[] = [
      {
        id: '1',
        title: 'Développement Web Fullstack - React, Node, MongoDB',
        category: 'dev',
        subcategory: 'Fullstack',
        price: 150,
        image: 'https://images.unsplash.com/photo-1587620962725-abab7fe55159?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1031&q=80',
        provider: {
          id: 'user1',
          name: 'David Martin',
          avatar: 'https://randomuser.me/api/portraits/men/32.jpg',
        },
        rating: 4.8,
        reviewCount: 24,
      },
      {
        id: '2',
        title: 'Logo et Identité Visuelle pour votre Entreprise',
        category: 'design',
        subcategory: 'Logo & Branding',
        price: 99,
        image: 'https://images.unsplash.com/photo-1611532736592-9f7d450fe20f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1374&q=80',
        provider: {
          id: 'user2',
          name: 'Sophie Dubois',
          avatar: 'https://randomuser.me/api/portraits/women/44.jpg',
        },
        rating: 4.9,
        reviewCount: 37,
      },
      {
        id: '3',
        title: 'Traduction Professionnelle en Français, Anglais et Espagnol',
        category: 'writing',
        subcategory: 'Traduction',
        price: 50,
        image: 'https://images.unsplash.com/photo-1456406644174-8ddd4cd52a06?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1368&q=80',
        provider: {
          id: 'user3',
          name: 'Pierre Lambert',
          avatar: 'https://randomuser.me/api/portraits/men/64.jpg',
        },
        rating: 4.7,
        reviewCount: 18,
      },
      {
        id: '4',
        title: 'Montage Vidéo Professionnel pour vos Projets',
        category: 'video',
        subcategory: 'Montage Vidéo',
        price: 120,
        image: 'https://images.unsplash.com/photo-1574717024453-354056de4834?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80',
        provider: {
          id: 'user4',
          name: 'Marie Leclerc',
          avatar: 'https://randomuser.me/api/portraits/women/33.jpg',
        },
        rating: 4.5,
        reviewCount: 12,
      },
      {
        id: '5',
        title: 'Création d\'Applications Mobiles iOS et Android',
        category: 'dev',
        subcategory: 'Mobile',
        price: 200,
        image: 'https://images.unsplash.com/photo-1551650975-87deedd944c3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1374&q=80',
        provider: {
          id: 'user5',
          name: 'Thomas Bernard',
          avatar: 'https://randomuser.me/api/portraits/men/76.jpg',
        },
        rating: 4.9,
        reviewCount: 8,
      },
      {
        id: '6',
        title: 'Rédaction Web SEO et Copywriting',
        category: 'writing',
        subcategory: 'Rédaction Web',
        price: 80,
        image: 'https://images.unsplash.com/photo-1560785496-3c9d27877182?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1374&q=80',
        provider: {
          id: 'user6',
          name: 'Julie Moreau',
          avatar: 'https://randomuser.me/api/portraits/women/17.jpg',
        },
        rating: 4.6,
        reviewCount: 15,
      },
      {
        id: '7',
        title: 'Création de Site WordPress Personnalisé',
        category: 'dev',
        subcategory: 'WordPress',
        price: 180,
        image: 'https://images.unsplash.com/photo-1559028012-481c04fa702d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1636&q=80',
        provider: {
          id: 'user7',
          name: 'Alexandre Petit',
          avatar: 'https://randomuser.me/api/portraits/men/22.jpg',
        },
        rating: 4.7,
        reviewCount: 19,
      },
      {
        id: '8',
        title: 'Gestion des Médias Sociaux pour votre Entreprise',
        category: 'marketing',
        subcategory: 'Médias Sociaux',
        price: 300,
        image: 'https://images.unsplash.com/photo-1611926653458-09294b3142bf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80',
        provider: {
          id: 'user8',
          name: 'Camille Rousseau',
          avatar: 'https://randomuser.me/api/portraits/women/57.jpg',
        },
        rating: 4.8,
        reviewCount: 28,
      },
    ];

    setServices(mockServices);
    setDisplayedServices(mockServices);
  }, []);

  useEffect(() => {
    // Filter services based on search, category, subcategory, and price range
    const filteredServices = services.filter(service => {
      const matchesSearch = service.title.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesCategory = selectedCategory === 'all' || service.category === selectedCategory;
      const matchesSubcategory = selectedSubcategory === 'all' || service.subcategory === selectedSubcategory;
      const matchesPrice = service.price >= priceRange[0] && service.price <= priceRange[1];

      return matchesSearch && matchesCategory && matchesSubcategory && matchesPrice;
    });

    setDisplayedServices(filteredServices);
  }, [searchTerm, selectedCategory, selectedSubcategory, priceRange, services]);

  const handleCategoryChange = (categoryId: string) => {
    setSelectedCategory(categoryId);
    setSelectedSubcategory('all'); // Reset subcategory when category changes
  };

  const handlePriceRangeChange = (min: number, max: number) => {
    setPriceRange([min, max]);
  };

  const getCategoryName = (categoryId: string) => {
    const category = categories.find(cat => cat.id === categoryId);
    return category ? category.name : 'Autre';
  };

  return (
    <div className="py-12 px-4">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Explorez nos Services
          </h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Trouvez des freelances talentueux proposant une multitude de services
            dans divers domaines pour répondre à tous vos besoins.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar - Filters */}
          <div className="lg:col-span-1">
            <div className="bg-workit-dark-card rounded-lg p-6 sticky top-6">
              <h2 className="text-xl font-semibold text-white mb-6">Filtres</h2>

              <div className="mb-6">
                <label htmlFor="search" className="block text-white text-sm font-medium mb-2">
                  Recherche
                </label>
                <input
                  type="text"
                  id="search"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Rechercher un service..."
                  className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                />
              </div>

              <div className="mb-6">
                <label className="block text-white text-sm font-medium mb-2">
                  Catégorie
                </label>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <input
                      type="radio"
                      id="cat-all"
                      name="category"
                      checked={selectedCategory === 'all'}
                      onChange={() => handleCategoryChange('all')}
                      className="h-4 w-4 border-gray-600 text-workit-purple focus:ring-workit-purple"
                    />
                    <label htmlFor="cat-all" className="ml-2 text-gray-300">
                      Toutes les catégories
                    </label>
                  </div>
                  {categories.map((category) => (
                    <div key={category.id} className="flex items-center">
                      <input
                        type="radio"
                        id={`cat-${category.id}`}
                        name="category"
                        checked={selectedCategory === category.id}
                        onChange={() => handleCategoryChange(category.id)}
                        className="h-4 w-4 border-gray-600 text-workit-purple focus:ring-workit-purple"
                      />
                      <label htmlFor={`cat-${category.id}`} className="ml-2 text-gray-300">
                        {category.name}
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              {selectedCategory !== 'all' && availableSubcategories.length > 0 && (
                <div className="mb-6">
                  <label className="block text-white text-sm font-medium mb-2">
                    Sous-catégorie
                  </label>
                  <div className="space-y-2">
                    <div className="flex items-center">
                      <input
                        type="radio"
                        id="subcat-all"
                        name="subcategory"
                        checked={selectedSubcategory === 'all'}
                        onChange={() => setSelectedSubcategory('all')}
                        className="h-4 w-4 border-gray-600 text-workit-purple focus:ring-workit-purple"
                      />
                      <label htmlFor="subcat-all" className="ml-2 text-gray-300">
                        Toutes les sous-catégories
                      </label>
                    </div>
                    {availableSubcategories.map((subcat, index) => (
                      <div key={index} className="flex items-center">
                        <input
                          type="radio"
                          id={`subcat-${index}`}
                          name="subcategory"
                          checked={selectedSubcategory === subcat}
                          onChange={() => setSelectedSubcategory(subcat)}
                          className="h-4 w-4 border-gray-600 text-workit-purple focus:ring-workit-purple"
                        />
                        <label htmlFor={`subcat-${index}`} className="ml-2 text-gray-300">
                          {subcat}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="mb-6">
                <label className="block text-white text-sm font-medium mb-2">
                  Fourchette de prix (TND)
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <input
                      type="number"
                      min="0"
                      value={priceRange[0]}
                      onChange={(e) => handlePriceRangeChange(Number(e.target.value), priceRange[1])}
                      className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                      placeholder="Min"
                    />
                  </div>
                  <div>
                    <input
                      type="number"
                      min={priceRange[0]}
                      value={priceRange[1]}
                      onChange={(e) => handlePriceRangeChange(priceRange[0], Number(e.target.value))}
                      className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                      placeholder="Max"
                    />
                  </div>
                </div>
              </div>

              <button
                onClick={() => {
                  setSearchTerm('');
                  setSelectedCategory('all');
                  setSelectedSubcategory('all');
                  setPriceRange([0, 1000]);
                }}
                className="w-full px-4 py-2 bg-gray-800 text-gray-300 rounded-md hover:bg-gray-700 transition"
              >
                Réinitialiser les filtres
              </button>
            </div>
          </div>

          {/* Main Content - Services */}
          <div className="lg:col-span-3">
            {displayedServices.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {displayedServices.map((service) => (
                  <div key={service.id} className="bg-workit-dark-card rounded-lg overflow-hidden shadow-lg hover:shadow-xl transition">
                    <div className="relative h-48 overflow-hidden">
                      <Link to={`/service/${service.id}`}>
                        <img
                          src={service.image}
                          alt={service.title}
                          className="w-full h-full object-cover transition duration-300 hover:scale-105"
                        />
                      </Link>
                      <div className="absolute top-2 left-2 bg-workit-purple text-white text-xs font-semibold px-2 py-1 rounded">
                        {getCategoryName(service.category)}
                      </div>
                      <button className="absolute top-2 right-2 bg-black bg-opacity-60 rounded-full p-2 hover:bg-opacity-80">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-5 w-5 text-white"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z" />
                        </svg>
                      </button>
                    </div>

                    <div className="p-4">
                      <div className="flex items-center mb-2">
                        <div className="flex-shrink-0 mr-2">
                          <div className="w-8 h-8 rounded-full overflow-hidden">
                            <img
                              src={service.provider.avatar}
                              alt={service.provider.name}
                              className="w-full h-full object-cover"
                            />
                          </div>
                        </div>
                        <span className="text-sm text-gray-400">{service.provider.name}</span>

                        {service.rating && (
                          <div className="ml-auto flex items-center">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-4 w-4 text-yellow-400"
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                            <span className="text-xs text-gray-400 ml-1">
                              {service.rating} ({service.reviewCount})
                            </span>
                          </div>
                        )}
                      </div>

                      <h3 className="font-semibold text-white mb-2 line-clamp-2">
                        <Link to={`/service/${service.id}`} className="hover:text-workit-purple">
                          {service.title}
                        </Link>
                      </h3>

                      <div className="flex justify-between items-center mt-4">
                        <div className="text-xs text-gray-400">à partir de:</div>
                        <div className="text-lg font-bold text-workit-purple">{service.price.toFixed(2)} TND</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="bg-workit-dark-card p-8 rounded-lg text-center">
                <h3 className="text-white text-xl mb-2">Aucun service trouvé</h3>
                <p className="text-gray-400">
                  Essayez d'ajuster vos filtres ou votre recherche pour trouver plus de résultats.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* CTA Section */}
        <div className="mt-12 bg-workit-purple p-8 rounded-lg text-center">
          <h2 className="text-2xl font-bold text-white mb-3">
            Vous avez une compétence à proposer?
          </h2>
          <p className="text-white opacity-90 mb-6 max-w-2xl mx-auto">
            Rejoignez notre communauté de freelances et commencez à proposer vos services
            dès aujourd'hui. C'est gratuit et facile!
          </p>
          <Link
            to="/register"
            className="inline-block bg-black text-white px-6 py-3 rounded-md font-medium hover:bg-gray-900 transition"
          >
            Devenir freelance
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ServicesPage;
