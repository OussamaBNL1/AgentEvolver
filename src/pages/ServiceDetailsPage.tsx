import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useUser } from '../context/UserContext';

interface Service {
  id: string;
  title: string;
  category: string;
  subcategory: string;
  price: number;
  image: string;
  description: string;
  deliveryTime: number;
  revisions: string;
  features: string[];
  provider: {
    id: string;
    name: string;
    avatar?: string;
    rating?: number;
    reviewCount?: number;
    joinedDate?: string;
    location?: string;
    description?: string;
  };
  gallery?: string[];
  faqs?: { question: string; answer: string }[];
  reviews?: {
    id: string;
    user: {
      name: string;
      avatar?: string;
    };
    rating: number;
    comment: string;
    date: string;
  }[];
}

const ServiceDetailsPage = () => {
  const { id } = useParams<{ id: string }>();
  const { isAuthenticated } = useUser();
  const [service, setService] = useState<Service | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState<'description' | 'reviews' | 'provider'>('description');

  useEffect(() => {
    // In a real app, this would be an API call to fetch the service details
    // For demo purposes, we're simulating a delay and using mock data
    const fetchService = async () => {
      setLoading(true);

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));

      // Mock service data based on id
      const mockService: Service = {
        id: id || '1',
        title: 'Développement Web Fullstack - React, Node, MongoDB',
        category: 'dev',
        subcategory: 'Fullstack',
        price: 150,
        image: 'https://images.unsplash.com/photo-1587620962725-abab7fe55159?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1031&q=80',
        description: 'Je développerai une application web complète en utilisant la stack MERN (MongoDB, Express, React, Node.js). Ce service comprend la conception, le développement et le déploiement d\'une application web responsive et moderne, adaptée à vos besoins spécifiques.',
        deliveryTime: 14,
        revisions: 'Illimitées',
        features: [
          'Conception d\'interface utilisateur',
          'Développement frontend avec React',
          'Développement backend avec Node.js et Express',
          'Base de données MongoDB',
          'API RESTful',
          'Authentification et autorisation',
          'Déploiement sur serveur',
        ],
        provider: {
          id: 'user1',
          name: 'David Martin',
          avatar: 'https://randomuser.me/api/portraits/men/32.jpg',
          rating: 4.8,
          reviewCount: 24,
          joinedDate: '2023-01-15',
          location: 'Paris, France',
          description: 'Développeur fullstack avec plus de 5 ans d\'expérience dans la création d\'applications web. Spécialisé dans React, Node.js et les bases de données NoSQL.',
        },
        gallery: [
          'https://images.unsplash.com/photo-1551434678-e076c223a692?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80',
          'https://images.unsplash.com/photo-1507238691740-187a5b1d37b8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1455&q=80',
          'https://images.unsplash.com/photo-1605379399642-870262d3d051?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1812&q=80',
        ],
        faqs: [
          {
            question: 'Quelles sont les technologies que vous utilisez?',
            answer: 'J\'utilise principalement la stack MERN (MongoDB, Express, React, Node.js) pour développer des applications web full-stack. Pour le frontend, j\'utilise React avec des bibliothèques comme Redux, React Router et Material-UI ou Tailwind CSS. Pour le backend, j\'utilise Node.js avec Express et MongoDB comme base de données.',
          },
          {
            question: 'Combien de temps faut-il pour développer une application web complète?',
            answer: 'Le délai de développement dépend de la complexité du projet. Pour une application web simple à moyenne, comptez environ 2 à 4 semaines. Pour des projets plus complexes, le délai peut s\'étendre à plusieurs mois. Je fournis toujours une estimation précise après avoir discuté des détails du projet.',
          },
          {
            question: 'Proposez-vous un support après la livraison?',
            answer: 'Oui, je propose un support technique gratuit pendant 1 mois après la livraison du projet. Ce support inclut la correction de bugs et l\'assistance pour les problèmes mineurs. Pour des modifications ou des fonctionnalités supplémentaires, un contrat de maintenance peut être négocié.',
          },
        ],
        reviews: [
          {
            id: 'rev1',
            user: {
              name: 'Jean Dupont',
              avatar: 'https://randomuser.me/api/portraits/men/1.jpg',
            },
            rating: 5,
            comment: 'David a réalisé un excellent travail pour notre site web. Le code est propre, bien structuré et le résultat final correspond parfaitement à nos attentes. Je recommande vivement ses services!',
            date: '2024-02-18',
          },
          {
            id: 'rev2',
            user: {
              name: 'Amélie Moreau',
              avatar: 'https://randomuser.me/api/portraits/women/2.jpg',
            },
            rating: 4,
            comment: 'Développeur très professionnel qui a su comprendre rapidement nos besoins. Le site est rapide et responsive. Seul petit bémol sur quelques détails d\'interface qui ont nécessité des ajustements, mais dans l\'ensemble, très satisfaite!',
            date: '2024-01-25',
          },
        ],
      };

      setService(mockService);
      setLoading(false);
    };

    fetchService();
  }, [id]);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-[50vh]">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-workit-purple"></div>
      </div>
    );
  }

  if (!service) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <h1 className="text-2xl font-bold text-white mb-4">Service non trouvé</h1>
        <p className="text-gray-400 mb-8">Le service que vous recherchez n'existe pas ou a été supprimé.</p>
        <Link to="/services" className="bg-workit-purple text-white px-6 py-2 rounded-md hover:bg-workit-purple-light transition">
          Retourner aux services
        </Link>
      </div>
    );
  }

  // Helper function to safely get provider rating
  const getProviderRating = () => {
    return service.provider.rating || 0;
  };

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content - Left Side */}
        <div className="lg:col-span-2">
          <h1 className="text-2xl md:text-3xl font-bold text-white mb-4">
            {service.title}
          </h1>

          <div className="flex items-center mb-6">
            <div className="w-10 h-10 rounded-full overflow-hidden mr-3">
              <img
                src={service.provider.avatar}
                alt={service.provider.name}
                className="w-full h-full object-cover"
              />
            </div>
            <div>
              <div className="text-white font-medium">
                {service.provider.name}
              </div>
              <div className="flex items-center text-sm">
                {service.provider.rating && (
                  <div className="flex items-center text-yellow-400 mr-2">
                    <div className="flex text-yellow-400">
                      {[...Array(5)].map((_, i) => (
                        <svg
                          key={i}
                          xmlns="http://www.w3.org/2000/svg"
                          className={`h-4 w-4 ${i < Math.floor(getProviderRating()) ? 'text-yellow-400' : 'text-gray-600'}`}
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                        </svg>
                      ))}
                      <span className="text-gray-400 ml-1">
                        {service.provider.rating}
                      </span>
                    </div>
                    <span className="text-gray-500 ml-2">
                      ({service.provider.reviewCount} avis)
                    </span>
                  </div>
                )}
                {service.provider.location && (
                  <span className="text-gray-400">
                    | {service.provider.location}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Service Gallery */}
          <div className="mb-8">
            <div className="rounded-lg overflow-hidden h-[400px]">
              <img
                src={service.image}
                alt={service.title}
                className="w-full h-full object-cover"
              />
            </div>

            {service.gallery && service.gallery.length > 0 && (
              <div className="grid grid-cols-3 gap-2 mt-2">
                {service.gallery.map((img, index) => (
                  <div key={index} className="h-24 rounded-lg overflow-hidden">
                    <img
                      src={img}
                      alt={`${service.title} - image ${index + 1}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Tabs */}
          <div className="mb-6 border-b border-gray-800">
            <div className="flex">
              <button
                onClick={() => setSelectedTab('description')}
                className={`px-4 py-2 font-medium text-sm ${
                  selectedTab === 'description'
                  ? 'text-workit-purple border-b-2 border-workit-purple'
                  : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                Description
              </button>

              <button
                onClick={() => setSelectedTab('reviews')}
                className={`px-4 py-2 font-medium text-sm ${
                  selectedTab === 'reviews'
                  ? 'text-workit-purple border-b-2 border-workit-purple'
                  : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                Avis ({service.reviews?.length || 0})
              </button>

              <button
                onClick={() => setSelectedTab('provider')}
                className={`px-4 py-2 font-medium text-sm ${
                  selectedTab === 'provider'
                  ? 'text-workit-purple border-b-2 border-workit-purple'
                  : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                À propos du vendeur
              </button>
            </div>
          </div>

          {/* Tab Content */}
          <div>
            {selectedTab === 'description' && (
              <div>
                <h2 className="text-xl font-semibold text-white mb-4">Description</h2>
                <p className="text-gray-400 mb-6">
                  {service.description}
                </p>

                <h3 className="text-lg font-semibold text-white mb-3">Ce qui est inclus</h3>
                <ul className="mb-6">
                  {service.features.map((feature, index) => (
                    <li key={index} className="flex items-start mb-2">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-5 w-5 text-green-500 mr-2 mt-0.5"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                      >
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      <span className="text-gray-300">{feature}</span>
                    </li>
                  ))}
                </ul>

                {service.faqs && service.faqs.length > 0 && (
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Questions Fréquentes</h3>
                    <div className="space-y-4 mb-6">
                      {service.faqs.map((faq, index) => (
                        <div key={index} className="bg-gray-800 bg-opacity-50 rounded-lg p-4">
                          <h4 className="text-white font-medium mb-2">{faq.question}</h4>
                          <p className="text-gray-400 text-sm">{faq.answer}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {selectedTab === 'reviews' && (
              <div>
                <h2 className="text-xl font-semibold text-white mb-4">Avis</h2>

                {service.reviews && service.reviews.length > 0 ? (
                  <div className="space-y-6">
                    {service.reviews.map((review) => (
                      <div key={review.id} className="border-b border-gray-800 pb-6">
                        <div className="flex items-start">
                          <div className="flex-shrink-0 mr-3">
                            <div className="w-10 h-10 rounded-full overflow-hidden">
                              <img
                                src={review.user.avatar}
                                alt={review.user.name}
                                className="w-full h-full object-cover"
                              />
                            </div>
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <h4 className="text-white font-medium">{review.user.name}</h4>
                              <span className="text-gray-500 text-sm">
                                {new Date(review.date).toLocaleDateString('fr-FR')}
                              </span>
                            </div>
                            <div className="flex text-yellow-400 mb-2">
                              {[...Array(5)].map((_, i) => (
                                <svg
                                  key={i}
                                  xmlns="http://www.w3.org/2000/svg"
                                  className={`h-4 w-4 ${i < review.rating ? 'text-yellow-400' : 'text-gray-600'}`}
                                  viewBox="0 0 20 20"
                                  fill="currentColor"
                                >
                                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                                </svg>
                              ))}
                            </div>
                            <p className="text-gray-400">{review.comment}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-400">Aucun avis pour ce service.</p>
                )}
              </div>
            )}

            {selectedTab === 'provider' && (
              <div>
                <h2 className="text-xl font-semibold text-white mb-4">À propos du vendeur</h2>

                <div className="flex items-center mb-6">
                  <div className="w-20 h-20 rounded-full overflow-hidden mr-4">
                    <img
                      src={service.provider.avatar}
                      alt={service.provider.name}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div>
                    <h3 className="text-white font-semibold text-lg">
                      {service.provider.name}
                    </h3>
                    <p className="text-gray-400 mb-1">
                      {service.provider.location}
                    </p>
                    {service.provider.rating && (
                      <div className="flex items-center text-sm">
                        <div className="flex text-yellow-400">
                          {[...Array(5)].map((_, i) => (
                            <svg
                              key={i}
                              xmlns="http://www.w3.org/2000/svg"
                              className={`h-4 w-4 ${i < Math.floor(getProviderRating()) ? 'text-yellow-400' : 'text-gray-600'}`}
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                          ))}
                          <span className="text-gray-400 ml-1">
                            {service.provider.rating}
                          </span>
                        </div>
                        <span className="text-gray-500 ml-2">
                          ({service.provider.reviewCount} avis)
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-gray-800 bg-opacity-50 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm mb-1">Membre depuis</div>
                    <div className="text-white">
                      {new Date(service.provider.joinedDate || '').toLocaleDateString('fr-FR')}
                    </div>
                  </div>
                  <div className="bg-gray-800 bg-opacity-50 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm mb-1">Délai de réponse</div>
                    <div className="text-white">Moins d'une heure</div>
                  </div>
                  <div className="bg-gray-800 bg-opacity-50 p-4 rounded-lg">
                    <div className="text-gray-400 text-sm mb-1">Dernière livraison</div>
                    <div className="text-white">Il y a 2 jours</div>
                  </div>
                </div>

                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-white mb-3">Description</h3>
                  <p className="text-gray-400">
                    {service.provider.description}
                  </p>
                </div>

                <Link
                  to={`/profile/${service.provider.id}`}
                  className="inline-block border border-workit-purple text-workit-purple px-4 py-2 rounded-md text-sm font-medium hover:bg-workit-purple hover:text-white transition"
                >
                  Voir le profil
                </Link>
              </div>
            )}
          </div>
        </div>

        {/* Sidebar - Right Side */}
        <div className="lg:col-span-1">
          <div className="bg-workit-dark-card rounded-lg p-6 sticky top-6">
            <div className="text-center mb-6">
              <div className="text-3xl font-bold text-workit-purple mb-4">
                {service.price.toFixed(2)} TND
              </div>
              <p className="text-gray-400 text-sm">
                Prix fixe pour ce service
              </p>
            </div>

            <div className="grid grid-cols-3 mb-6 text-center">
              <div>
                <h4 className="text-sm text-gray-400 mb-1">Délai de livraison</h4>
                <p className="text-white font-semibold">{service.deliveryTime} jours</p>
              </div>
              <div>
                <h4 className="text-sm text-gray-400 mb-1">Révisions</h4>
                <p className="text-white font-semibold">{service.revisions}</p>
              </div>
              <div>
                <h4 className="text-sm text-gray-400 mb-1">Commandes en cours</h4>
                <p className="text-white font-semibold">3</p>
              </div>
            </div>

            <div className="space-y-3 mb-6">
              {service.features.slice(0, 5).map((feature, index) => (
                <div key={index} className="flex items-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5 text-green-500 mr-2"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span className="text-gray-300 text-sm">{feature}</span>
                </div>
              ))}
            </div>

            <button
              onClick={() => {
                if (!isAuthenticated) {
                  // Redirect to login or show login modal
                  alert('Veuillez vous connecter pour commander ce service.');
                } else {
                  // Navigate to checkout or add to cart
                  alert('Service ajouté au panier!');
                }
              }}
              className="w-full bg-workit-purple text-white py-3 rounded-md font-medium hover:bg-workit-purple-light transition mb-4"
            >
              Commander ce service
            </button>

            <button
              className="w-full bg-transparent border border-workit-purple text-workit-purple py-3 rounded-md font-medium hover:bg-workit-purple hover:text-white transition"
            >
              Contacter le vendeur
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ServiceDetailsPage;
