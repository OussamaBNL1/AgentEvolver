import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useUser } from '../../context/UserContext';
import {
  Search,
  Filter,
  ArrowRight,
  ArrowUp,
  ArrowDown,
  Calendar,
  Package,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Download
} from 'lucide-react';

interface Order {
  id: string;
  client: {
    id: string;
    name: string;
    avatar?: string;
  };
  service: {
    id: string;
    title: string;
    price: number;
  };
  createdAt: string;
  deadline: string;
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled' | 'revision';
  isPaid: boolean;
  notes?: string;
}

const DashboardOrders = () => {
  const [orders, setOrders] = useState<Order[]>([]);
  const [filteredOrders, setFilteredOrders] = useState<Order[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'active' | 'completed' | 'cancelled'>('all');
  const [sortBy, setSortBy] = useState<'date' | 'deadline' | 'price'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    // Simulate API call to get user's orders
    const fetchOrders = async () => {
      setIsLoading(true);

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 800));

      // Mock orders data
      const mockOrders: Order[] = [
        {
          id: 'ORD-001',
          client: {
            id: 'client-1',
            name: 'Jean Dupont',
            avatar: 'https://randomuser.me/api/portraits/men/1.jpg',
          },
          service: {
            id: 'service-1',
            title: 'Développement Web Fullstack',
            price: 150,
          },
          createdAt: '2025-03-18',
          deadline: '2025-04-01',
          status: 'in_progress',
          isPaid: true,
          notes: 'Client demande une attention particulière sur le design.',
        },
        {
          id: 'ORD-002',
          client: {
            id: 'client-2',
            name: 'Sophie Martin',
            avatar: 'https://randomuser.me/api/portraits/women/2.jpg',
          },
          service: {
            id: 'service-2',
            title: 'Création de Logo',
            price: 80,
          },
          createdAt: '2025-03-15',
          deadline: '2025-03-22',
          status: 'completed',
          isPaid: true,
        },
        {
          id: 'ORD-003',
          client: {
            id: 'client-3',
            name: 'Marc Lefevre',
            avatar: 'https://randomuser.me/api/portraits/men/3.jpg',
          },
          service: {
            id: 'service-1',
            title: 'Développement Web Fullstack',
            price: 150,
          },
          createdAt: '2025-03-12',
          deadline: '2025-03-26',
          status: 'pending',
          isPaid: false,
        },
        {
          id: 'ORD-004',
          client: {
            id: 'client-4',
            name: 'Laura Blanc',
            avatar: 'https://randomuser.me/api/portraits/women/4.jpg',
          },
          service: {
            id: 'service-3',
            title: 'Développement d\'API',
            price: 120,
          },
          createdAt: '2025-03-10',
          deadline: '2025-03-24',
          status: 'cancelled',
          isPaid: false,
          notes: 'Annulé à la demande du client.',
        },
        {
          id: 'ORD-005',
          client: {
            id: 'client-5',
            name: 'Thomas Dubois',
            avatar: 'https://randomuser.me/api/portraits/men/5.jpg',
          },
          service: {
            id: 'service-2',
            title: 'Création de Logo',
            price: 80,
          },
          createdAt: '2025-03-08',
          deadline: '2025-03-15',
          status: 'revision',
          isPaid: true,
          notes: 'Client a demandé des modifications sur les couleurs.',
        },
      ];

      setOrders(mockOrders);
      setFilteredOrders(mockOrders);
      setIsLoading(false);
    };

    fetchOrders();
  }, []);

  // Filter and sort orders
  useEffect(() => {
    let filtered = [...orders];

    // Apply status filter
    if (statusFilter !== 'all') {
      if (statusFilter === 'active') {
        filtered = filtered.filter(order =>
          ['pending', 'in_progress', 'revision'].includes(order.status)
        );
      } else if (statusFilter === 'completed') {
        filtered = filtered.filter(order => order.status === 'completed');
      } else if (statusFilter === 'cancelled') {
        filtered = filtered.filter(order => order.status === 'cancelled');
      }
    }

    // Apply search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(
        order =>
          order.id.toLowerCase().includes(term) ||
          order.client.name.toLowerCase().includes(term) ||
          order.service.title.toLowerCase().includes(term)
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      if (sortBy === 'date') {
        return sortOrder === 'asc'
          ? new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
          : new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      }

      if (sortBy === 'deadline') {
        return sortOrder === 'asc'
          ? new Date(a.deadline).getTime() - new Date(b.deadline).getTime()
          : new Date(b.deadline).getTime() - new Date(a.deadline).getTime();
      }

      if (sortBy === 'price') {
        return sortOrder === 'asc'
          ? a.service.price - b.service.price
          : b.service.price - a.service.price;
      }

      return 0;
    });

    setFilteredOrders(filtered);
  }, [searchTerm, statusFilter, sortBy, sortOrder, orders]);

  // Helper function to toggle sort order
  const toggleSort = (field: 'date' | 'deadline' | 'price') => {
    if (sortBy === field) {
      setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  // Helper function to get status badge
  const getStatusBadge = (status: Order['status']) => {
    switch (status) {
      case 'pending':
        return (
          <div className="flex items-center text-yellow-500 bg-yellow-500/10 px-2 py-1 rounded-full text-xs font-medium">
            <Clock size={14} className="mr-1" />
            En attente
          </div>
        );
      case 'in_progress':
        return (
          <div className="flex items-center text-blue-500 bg-blue-500/10 px-2 py-1 rounded-full text-xs font-medium">
            <Package size={14} className="mr-1" />
            En cours
          </div>
        );
      case 'completed':
        return (
          <div className="flex items-center text-green-500 bg-green-500/10 px-2 py-1 rounded-full text-xs font-medium">
            <CheckCircle size={14} className="mr-1" />
            Terminé
          </div>
        );
      case 'cancelled':
        return (
          <div className="flex items-center text-red-500 bg-red-500/10 px-2 py-1 rounded-full text-xs font-medium">
            <XCircle size={14} className="mr-1" />
            Annulé
          </div>
        );
      case 'revision':
        return (
          <div className="flex items-center text-orange-500 bg-orange-500/10 px-2 py-1 rounded-full text-xs font-medium">
            <AlertTriangle size={14} className="mr-1" />
            Révision
          </div>
        );
      default:
        return null;
    }
  };

  // Helper function to calculate days left
  const getDaysLeft = (deadline: string) => {
    const today = new Date();
    const deadlineDate = new Date(deadline);
    const diffTime = deadlineDate.getTime() - today.getTime();
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    return diffDays;
  };

  // Helper function to format dates
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('fr-FR');
  };

  return (
    <div>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Commandes</h1>
          <p className="text-gray-400">
            Gérez et suivez toutes vos commandes en cours et passées.
          </p>
        </div>
        <div className="mt-4 md:mt-0">
          <button
            className="inline-flex items-center bg-gray-800 text-white px-4 py-2 rounded-md hover:bg-gray-700 transition"
          >
            <Download size={16} className="mr-2" />
            Exporter
          </button>
        </div>
      </div>

      {/* Filters and Search */}
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
              placeholder="Rechercher une commande, un client..."
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
              Toutes
            </button>
            <button
              onClick={() => setStatusFilter('active')}
              className={`px-4 py-2 rounded-md text-sm ${
                statusFilter === 'active'
                  ? 'bg-blue-700 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Actives
            </button>
            <button
              onClick={() => setStatusFilter('completed')}
              className={`px-4 py-2 rounded-md text-sm ${
                statusFilter === 'completed'
                  ? 'bg-green-700 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Terminées
            </button>
            <button
              onClick={() => setStatusFilter('cancelled')}
              className={`px-4 py-2 rounded-md text-sm ${
                statusFilter === 'cancelled'
                  ? 'bg-red-700 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Annulées
            </button>
          </div>
        </div>
      </div>

      {/* Orders Table */}
      {isLoading ? (
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-workit-purple"></div>
        </div>
      ) : filteredOrders.length > 0 ? (
        <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-800">
                  <th className="py-4 px-6 font-medium">Commande</th>
                  <th className="py-4 px-6 font-medium">Client</th>
                  <th className="py-4 px-6 font-medium">Service</th>
                  <th className="py-4 px-6 font-medium">
                    <div className="flex items-center cursor-pointer" onClick={() => toggleSort('date')}>
                      <span>Date</span>
                      {sortBy === 'date' && (
                        <span className="ml-1">
                          {sortOrder === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
                        </span>
                      )}
                    </div>
                  </th>
                  <th className="py-4 px-6 font-medium">
                    <div className="flex items-center cursor-pointer" onClick={() => toggleSort('deadline')}>
                      <span>Échéance</span>
                      {sortBy === 'deadline' && (
                        <span className="ml-1">
                          {sortOrder === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
                        </span>
                      )}
                    </div>
                  </th>
                  <th className="py-4 px-6 font-medium">Statut</th>
                  <th className="py-4 px-6 font-medium">
                    <div className="flex items-center cursor-pointer" onClick={() => toggleSort('price')}>
                      <span>Montant</span>
                      {sortBy === 'price' && (
                        <span className="ml-1">
                          {sortOrder === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
                        </span>
                      )}
                    </div>
                  </th>
                  <th className="py-4 px-6 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredOrders.map((order) => {
                  const daysLeft = getDaysLeft(order.deadline);
                  const isLate = daysLeft < 0 && order.status !== 'completed' && order.status !== 'cancelled';

                  return (
                    <tr key={order.id} className="border-b border-gray-800 hover:bg-gray-800/50 transition">
                      <td className="py-4 px-6">
                        <Link
                          to={`/dashboard/orders/${order.id}`}
                          className="text-workit-purple hover:underline"
                        >
                          {order.id}
                        </Link>
                      </td>
                      <td className="py-4 px-6">
                        <div className="flex items-center">
                          <div className="w-8 h-8 rounded-full overflow-hidden mr-3">
                            <img
                              src={order.client.avatar}
                              alt={order.client.name}
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <span className="text-white">{order.client.name}</span>
                        </div>
                      </td>
                      <td className="py-4 px-6 text-gray-300">{order.service.title}</td>
                      <td className="py-4 px-6 text-gray-400">
                        {formatDate(order.createdAt)}
                      </td>
                      <td className="py-4 px-6">
                        <div className="flex items-center">
                          <Calendar size={14} className="text-gray-500 mr-1" />
                          <span
                            className={`${
                              isLate ? 'text-red-400' : 'text-gray-400'
                            }`}
                          >
                            {formatDate(order.deadline)}
                          </span>
                          {daysLeft > 0 && order.status !== 'completed' && order.status !== 'cancelled' && (
                            <span className="ml-2 text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded-full">
                              {daysLeft} jour{daysLeft > 1 ? 's' : ''}
                            </span>
                          )}
                          {isLate && (
                            <span className="ml-2 text-xs bg-red-900/50 text-red-400 px-2 py-0.5 rounded-full">
                              En retard
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        {getStatusBadge(order.status)}
                        {order.isPaid && (
                          <div className="mt-1 text-xs text-gray-400">
                            Payé
                          </div>
                        )}
                      </td>
                      <td className="py-4 px-6 text-white font-medium">
                        {order.service.price.toFixed(2)} TND
                      </td>
                      <td className="py-4 px-6 text-right">
                        <Link
                          to={`/dashboard/orders/${order.id}`}
                          className="inline-flex items-center text-workit-purple hover:text-workit-purple-light"
                        >
                          <span className="mr-1">Détails</span>
                          <ArrowRight size={14} />
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-8 text-center">
          <div className="text-gray-400 mb-4">
            Aucune commande ne correspond à vos critères de recherche.
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
    </div>
  );
};

export default DashboardOrders;
