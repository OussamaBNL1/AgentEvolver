import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useUser } from '../../context/UserContext';
import { useMessages } from '../../context/MessageContext';
import {
  Clock,
  Check,
  X,
  MessageSquare,
  FileText,
  Calendar,
  DollarSign,
  User,
  ChevronRight
} from 'lucide-react';

interface Order {
  id: string;
  serviceId: string;
  serviceName: string;
  clientId: string;
  clientName: string;
  clientAvatar?: string;
  date: string;
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled';
  price: number;
  requirements?: string;
  deliveryDate?: string;
}

const DashboardOrderDetails = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useUser();
  const { createConversation } = useMessages();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [order, setOrder] = useState<Order | null>(null);

  useEffect(() => {
    // Simulating fetching order data
    const fetchOrder = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // In a real application, this would be an API call
        await new Promise(resolve => setTimeout(resolve, 500)); // Simulate network delay

        // Mock order data for demo purposes
        const mockOrder: Order = {
          id: id || 'ORD-001',
          serviceId: 'svc_1',
          serviceName: 'Développement Web Fullstack',
          clientId: 'client_1',
          clientName: 'Jean Dupont',
          clientAvatar: 'https://randomuser.me/api/portraits/men/1.jpg',
          date: '2025-03-18',
          status: 'in_progress',
          price: 150,
          requirements: 'J\'ai besoin d\'un site e-commerce avec une page d\'accueil, des pages de produits, un panier et un système de paiement sécurisé.',
          deliveryDate: '2025-04-02',
        };

        setOrder(mockOrder);
      } catch (err) {
        setError('Error loading order data');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchOrder();
  }, [id]);

  const handleStatusChange = async (newStatus: Order['status']) => {
    if (!order) return;

    // In a real application, this would be an API call
    // Simulating updating order status
    setOrder(prev => prev ? { ...prev, status: newStatus } : null);
  };

  const handleMessageClient = async () => {
    if (!order) return;

    try {
      // Create or open conversation with the client
      const conversationId = await createConversation(order.clientId, `Order: ${order.id}`);
      navigate(`/dashboard/messages/${conversationId}`);
    } catch (error) {
      console.error('Error creating conversation:', error);
    }
  };

  // Helper function to get status styling
  const getStatusStyle = (status: Order['status']) => {
    switch (status) {
      case 'pending':
        return 'bg-yellow-500/10 text-yellow-500';
      case 'in_progress':
        return 'bg-blue-500/10 text-blue-500';
      case 'completed':
        return 'bg-green-500/10 text-green-500';
      case 'cancelled':
        return 'bg-red-500/10 text-red-500';
      default:
        return 'bg-gray-500/10 text-gray-500';
    }
  };

  // Helper function to get status label
  const getStatusLabel = (status: Order['status']) => {
    switch (status) {
      case 'pending':
        return 'En attente';
      case 'in_progress':
        return 'En cours';
      case 'completed':
        return 'Terminé';
      case 'cancelled':
        return 'Annulé';
      default:
        return status;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-workit-purple"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 text-red-500 p-4 rounded-md mb-6">
        <p>{error}</p>
      </div>
    );
  }

  if (!order) {
    return (
      <div className="bg-yellow-500/10 text-yellow-500 p-4 rounded-md mb-6">
        <p>Commande non trouvée</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <div className="flex items-center space-x-2 text-gray-400 mb-2">
            <Link to="/dashboard/orders" className="hover:text-white transition">
              Commandes
            </Link>
            <ChevronRight size={16} />
            <span className="text-white">{order.id}</span>
          </div>
          <h1 className="text-2xl font-bold text-white">{order.serviceName}</h1>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={handleMessageClient}
            className="flex items-center px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-600 transition"
          >
            <MessageSquare size={18} className="mr-2" />
            Message
          </button>
          <button
            onClick={() => navigate('/dashboard/orders')}
            className="px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-600 transition"
          >
            Retour
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Order Info */}
        <div className="lg:col-span-2">
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h2 className="text-xl font-semibold text-white mb-1">Détails de la commande</h2>
                <p className="text-gray-400">Créée le {new Date(order.date).toLocaleDateString('fr-FR')}</p>
              </div>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusStyle(order.status)}`}>
                {getStatusLabel(order.status)}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div className="flex items-center">
                <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center mr-3">
                  <Calendar size={20} className="text-workit-purple" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Date de commande</p>
                  <p className="text-white font-medium">
                    {new Date(order.date).toLocaleDateString('fr-FR')}
                  </p>
                </div>
              </div>

              <div className="flex items-center">
                <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center mr-3">
                  <DollarSign size={20} className="text-workit-purple" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Montant</p>
                  <p className="text-white font-medium">{order.price.toFixed(2)} TND</p>
                </div>
              </div>

              <div className="flex items-center">
                <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center mr-3">
                  <User size={20} className="text-workit-purple" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Client</p>
                  <p className="text-white font-medium">{order.clientName}</p>
                </div>
              </div>

              <div className="flex items-center">
                <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center mr-3">
                  <Clock size={20} className="text-workit-purple" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Date de livraison</p>
                  <p className="text-white font-medium">
                    {order.deliveryDate ? new Date(order.deliveryDate).toLocaleDateString('fr-FR') : 'Non définie'}
                  </p>
                </div>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-lg font-medium text-white mb-3">Exigences du client</h3>
              <div className="bg-gray-800 p-4 rounded-md text-gray-300">
                {order.requirements || 'Aucune exigence spécifiée'}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium text-white mb-3">Modifier le statut</h3>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => handleStatusChange('pending')}
                  className={`flex items-center px-4 py-2 rounded-md transition ${
                    order.status === 'pending'
                      ? 'bg-yellow-500/20 text-yellow-500 border border-yellow-500'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  <Clock size={18} className="mr-2" />
                  En attente
                </button>
                <button
                  onClick={() => handleStatusChange('in_progress')}
                  className={`flex items-center px-4 py-2 rounded-md transition ${
                    order.status === 'in_progress'
                      ? 'bg-blue-500/20 text-blue-500 border border-blue-500'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  <FileText size={18} className="mr-2" />
                  En cours
                </button>
                <button
                  onClick={() => handleStatusChange('completed')}
                  className={`flex items-center px-4 py-2 rounded-md transition ${
                    order.status === 'completed'
                      ? 'bg-green-500/20 text-green-500 border border-green-500'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  <Check size={18} className="mr-2" />
                  Terminé
                </button>
                <button
                  onClick={() => handleStatusChange('cancelled')}
                  className={`flex items-center px-4 py-2 rounded-md transition ${
                    order.status === 'cancelled'
                      ? 'bg-red-500/20 text-red-500 border border-red-500'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  <X size={18} className="mr-2" />
                  Annulé
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Client Info */}
        <div>
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
            <h2 className="text-xl font-semibold text-white mb-4">Informations du client</h2>

            <div className="flex items-center mb-4">
              <div className="w-12 h-12 rounded-full overflow-hidden mr-3">
                {order.clientAvatar ? (
                  <img
                    src={order.clientAvatar}
                    alt={order.clientName}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-gray-700 flex items-center justify-center text-white">
                    {order.clientName.charAt(0).toUpperCase()}
                  </div>
                )}
              </div>
              <div>
                <p className="text-white font-medium">{order.clientName}</p>
                <p className="text-gray-400 text-sm">Client depuis 2025</p>
              </div>
            </div>

            <div className="space-y-4 mb-6">
              <div className="bg-gray-800 p-3 rounded-md">
                <p className="text-sm text-gray-400">Email</p>
                <p className="text-white">client@example.com</p>
              </div>
              <div className="bg-gray-800 p-3 rounded-md">
                <p className="text-sm text-gray-400">Localisation</p>
                <p className="text-white">Tunis, Tunisie</p>
              </div>
              <div className="bg-gray-800 p-3 rounded-md">
                <p className="text-sm text-gray-400">Commandes précédentes</p>
                <p className="text-white">2 commandes</p>
              </div>
            </div>

            <button
              onClick={handleMessageClient}
              className="w-full flex items-center justify-center px-4 py-2 bg-workit-purple text-white rounded-md hover:bg-workit-purple-light transition"
            >
              <MessageSquare size={18} className="mr-2" />
              Contacter le client
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardOrderDetails;
