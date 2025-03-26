import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useUser } from '../../context/UserContext';
import {
  Plus,
  Search,
  Edit,
  Trash2,
  Eye,
  MoreVertical,
  AlertCircle,
  CheckCircle2,
  Star
} from 'lucide-react';

interface Service {
  id: string;
  title: string;
  category: string;
  subcategory: string;
  price: number;
  image: string;
  description: string;
  status: 'active' | 'pending' | 'paused';
  createdAt: string;
  rating?: number;
  orders: number;
  views: number;
}

const DashboardServices = () => {
  const { user } = useUser();
  const [services, setServices] = useState<Service[]>([]);
  const [filteredServices, setFilteredServices] = useState<Service[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'active' | 'pending' | 'paused'>('all');
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [serviceToDelete, setServiceToDelete] = useState<Service | null>(null);

  useEffect(() => {
    // Simulate API call to get user's services
    const fetchServices = async () => {
      setIsLoading(true);

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 800));

      // Mock services data
      const mockServices: Service[] = [
        {
          id: '1',
          title: 'Développement Web Fullstack - React, Node, MongoDB',
          category: 'Développement',
          subcategory: 'Développement Web',
          price: 150,
          image: 'https://images.unsplash.com/photo-1587620962725-abab7fe55159?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1031&q=80',
          description: 'Je développerai une application web complète en utilisant la stack MERN (MongoDB, Express, React, Node.js).',
          status: 'active',
          createdAt: '2025-02-10',
          rating: 4.8,
          orders: 14,
          views: 243,
        },
        {
          id: '2',
          title: 'Développement d\'API RESTful avec Node.js et Express',
          category: 'Développement',
          subcategory: 'API & Backend',
          price: 120,
          image: 'https://images.unsplash.com/photo-1555066931-4365d14bab8c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80',
          description: 'Je créerai une API RESTful complète avec Node.js et Express, incluant l\'authentification JWT et la documentation Swagger.',
          status: 'active',
          createdAt: '2025-02-15',
          rating: 4.6,
          orders: 7,
          views: 129,
        },
        {
          id: '3',
          title: 'Intégration de paiement Stripe dans une application React',
          category: 'Développement',
          subcategory: 'E-commerce',
          price: 100,
          image: 'https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80',
          description: 'J\'intégrerai la passerelle de paiement Stripe dans votre application React pour permettre les paiements par carte bancaire.',
          status: 'pending',
          createdAt: '2025-03-05',
          orders: 0,
          views: 42,
        },
        {
          id: '4',
          title: 'Création de Site WordPress Personnalisé',
          category: 'Développement',
          subcategory: 'WordPress',
          price: 180,
          image: 'https://images.unsplash.com/photo-1559028012-481c04fa702d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1636&q=80',
          description: 'Je créerai un site WordPress entièrement personnalisé avec un thème sur mesure et des fonctionnalités adaptées à vos besoins.',
          status: 'paused',
          createdAt: '2025-01-20',
          rating: 4.7,
          orders: 5,
          views: 87,
        },
      ];

      setServices(mockServices);
      setFilteredServices(mockServices);
      setIsLoading(false);
    };

    fetchServices();
  }, []);

  // Filter services based on search term and status filter
  useEffect(() => {
    let filtered = services;

    // Apply status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(service => service.status === statusFilter);
    }

    // Apply search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(
        service =>
          service.title.toLowerCase().includes(term) ||
          service.description.toLowerCase().includes(term) ||
          service.category.toLowerCase().includes(term) ||
          service.subcategory.toLowerCase().includes(term)
      );
    }

    setFilteredServices(filtered);
  }, [searchTerm, statusFilter, services]);

  const handleDeleteClick = (service: Service) => {
    setServiceToDelete(service);
    setShowDeleteModal(true);
  };

  const confirmDelete = async () => {
    if (!serviceToDelete) return;

    // Simulate API call to delete service
    setIsLoading(true);

    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));

    // Remove from local state
    setServices(prevServices =>
      prevServices.filter(s => s.id !== serviceToDelete.id)
    );

    setShowDeleteModal(false);
    setServiceToDelete(null);
    setIsLoading(false);
  };

  const getStatusChip = (status: Service['status']) => {
    switch (status) {
      case 'active':
        return (
          <div className="flex items-center text-green-500 bg-green-500/10 px-2 py-1 rounded-full text-xs font-medium">
            <CheckCircle2 size={14} className="mr-1" />
            Actif
          </div>
        );
      case 'pending':
        return (
          <div className="flex items-center text-yellow-500 bg-yellow-500/10 px-2 py-1 rounded-full text-xs font-medium">
            <AlertCircle size={14} className="mr-1" />
            En attente
          </div>
        );
      case 'paused':
        return (
          <div className="flex items-center text-gray-500 bg-gray-500/10 px-2 py-1 rounded-full text-xs font-medium">
            <AlertCircle size={14} className="mr-1" />
            Suspendu
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Mes services</h1>
          <p className="text-gray-400">
            Gérez vos services proposés aux clients sur WorkiT.
          </p>
        </div>
        <div className="mt-4 md:mt-0">
          <Link
            to="/dashboard/services/add"
            className="inline-flex items-center bg-workit-purple text-white px-4 py-2 rounded-md hover:bg-workit-purple-light transition"
          >
            <Plus size={18} className="mr-2" />
            Ajouter un service
          </Link>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-800 mb-6">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="relative flex-1">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search size={18} className="text-gray-500" />
            </div>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Rechercher un service..."
              className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
            />
          </div>

          <div className="flex space-x-2">
            <button
              onClick={() => setStatusFilter('all')}
              className={`px-4 py-2 rounded-md text-sm ${
                statusFilter === 'all'
                  ? 'bg-workit-purple text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Tous
            </button>
            <button
              onClick={() => setStatusFilter('active')}
              className={`px-4 py-2 rounded-md text-sm ${
                statusFilter === 'active'
                  ? 'bg-green-700 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Actifs
            </button>
            <button
              onClick={() => setStatusFilter('pending')}
              className={`px-4 py-2 rounded-md text-sm ${
                statusFilter === 'pending'
                  ? 'bg-yellow-700 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              En attente
            </button>
            <button
              onClick={() => setStatusFilter('paused')}
              className={`px-4 py-2 rounded-md text-sm ${
                statusFilter === 'paused'
                  ? 'bg-gray-700 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Suspendus
            </button>
          </div>
        </div>
      </div>

      {/* Services List */}
      {isLoading ? (
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-workit-purple"></div>
        </div>
      ) : filteredServices.length > 0 ? (
        <div className="grid grid-cols-1 gap-4">
          {filteredServices.map(service => (
            <div
              key={service.id}
              className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden hover:border-gray-700 transition"
            >
              <div className="flex flex-col md:flex-row">
                <div className="md:w-48 h-48 md:h-auto overflow-hidden">
                  <img
                    src={service.image}
                    alt={service.title}
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="p-6 flex-1">
                  <div className="flex flex-col md:flex-row justify-between">
                    <div>
                      <div className="flex items-center mb-2">
                        {getStatusChip(service.status)}
                        <span className="text-gray-400 text-sm ml-3">
                          Créé le {new Date(service.createdAt).toLocaleDateString('fr-FR')}
                        </span>
                      </div>

                      <h3 className="text-lg font-semibold text-white mb-2">
                        {service.title}
                      </h3>

                      <div className="text-gray-400 text-sm mb-3">
                        {service.category} &gt; {service.subcategory}
                      </div>

                      <p className="text-gray-300 mb-4 line-clamp-2">
                        {service.description}
                      </p>
                    </div>

                    <div className="md:text-right mt-4 md:mt-0">
                      <div className="text-xl font-bold text-workit-purple mb-2">
                        {service.price.toFixed(2)} TND
                      </div>

                      {service.rating && (
                        <div className="flex items-center mb-2 md:justify-end">
                          <Star size={16} className="text-yellow-500 mr-1" />
                          <span className="text-gray-300 text-sm">
                            {service.rating} ({service.orders} commandes)
                          </span>
                        </div>
                      )}

                      <div className="text-gray-400 text-sm">
                        {service.views} vues
                      </div>
                    </div>
                  </div>

                  <div className="flex mt-4 pt-4 border-t border-gray-800">
                    <Link
                      to={`/service/${service.id}`}
                      className="text-gray-400 hover:text-white flex items-center mr-4"
                    >
                      <Eye size={16} className="mr-1" />
                      <span>Voir</span>
                    </Link>
                    <Link
                      to={`/dashboard/services/edit/${service.id}`}
                      className="text-workit-purple hover:text-workit-purple-light flex items-center mr-4"
                    >
                      <Edit size={16} className="mr-1" />
                      <span>Modifier</span>
                    </Link>
                    <button
                      onClick={() => handleDeleteClick(service)}
                      className="text-red-500 hover:text-red-400 flex items-center"
                    >
                      <Trash2 size={16} className="mr-1" />
                      <span>Supprimer</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-8 text-center">
          <div className="text-gray-400 mb-4">
            Aucun service ne correspond à vos critères de recherche.
          </div>
          <button
            onClick={() => {
              setSearchTerm('');
              setStatusFilter('all');
            }}
            className="text-workit-purple hover:underline"
          >
            Réinitialiser les filtres
          </button>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteModal && serviceToDelete && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 rounded-lg p-6 max-w-md w-full border border-gray-800">
            <h3 className="text-xl font-semibold text-white mb-4">Confirmer la suppression</h3>
            <p className="text-gray-300 mb-6">
              Êtes-vous sûr de vouloir supprimer le service <span className="font-medium text-white">{serviceToDelete.title}</span> ? Cette action est irréversible.
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowDeleteModal(false)}
                className="px-4 py-2 bg-gray-800 text-gray-300 rounded-md hover:bg-gray-700 transition"
              >
                Annuler
              </button>
              <button
                onClick={confirmDelete}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition"
              >
                Supprimer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DashboardServices;
