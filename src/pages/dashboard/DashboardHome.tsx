import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useUser } from '../../context/UserContext';
import {
  TrendingUp,
  DollarSign,
  ShoppingBag,
  Users,
  ArrowUpRight,
  Clock,
  FileText,
  PlusCircle,
  MessageSquare,
  User
} from 'lucide-react';

interface DashboardStat {
  title: string;
  value: string | number;
  change: number;
  icon: JSX.Element;
  iconBg: string;
}

interface RecentOrder {
  id: string;
  clientName: string;
  clientAvatar?: string;
  service: string;
  date: string;
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled';
  price: number;
}

const DashboardHome = () => {
  const { user } = useUser();
  const [stats, setStats] = useState<DashboardStat[]>([]);
  const [recentOrders, setRecentOrders] = useState<RecentOrder[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate API call to get dashboard data
    const fetchDashboardData = async () => {
      setIsLoading(true);

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));

      // Mock dashboard stats
      const mockStats: DashboardStat[] = [
        {
          title: 'Ventes ce mois',
          value: '1,285 TND',
          change: 12.5,
          icon: <TrendingUp size={20} />,
          iconBg: 'bg-green-500',
        },
        {
          title: 'En attente',
          value: '450 TND',
          change: -3.2,
          icon: <Clock size={20} />,
          iconBg: 'bg-orange-500',
        },
        {
          title: 'Commandes',
          value: 14,
          change: 8.3,
          icon: <ShoppingBag size={20} />,
          iconBg: 'bg-purple-500',
        },
        {
          title: 'Nouveaux clients',
          value: 7,
          change: 4.6,
          icon: <Users size={20} />,
          iconBg: 'bg-blue-500',
        },
      ];

      // Mock recent orders
      const mockRecentOrders: RecentOrder[] = [
        {
          id: 'ORD-001',
          clientName: 'Jean Dupont',
          clientAvatar: 'https://randomuser.me/api/portraits/men/1.jpg',
          service: 'Développement Web Fullstack',
          date: '2025-03-18',
          status: 'in_progress',
          price: 150,
        },
        {
          id: 'ORD-002',
          clientName: 'Sophie Martin',
          clientAvatar: 'https://randomuser.me/api/portraits/women/2.jpg',
          service: 'Création de Logo',
          date: '2025-03-15',
          status: 'completed',
          price: 80,
        },
        {
          id: 'ORD-003',
          clientName: 'Marc Lefevre',
          clientAvatar: 'https://randomuser.me/api/portraits/men/3.jpg',
          service: 'Développement Web Fullstack',
          date: '2025-03-12',
          status: 'pending',
          price: 150,
        },
        {
          id: 'ORD-004',
          clientName: 'Laura Blanc',
          clientAvatar: 'https://randomuser.me/api/portraits/women/4.jpg',
          service: 'Développement d\'API',
          date: '2025-03-10',
          status: 'cancelled',
          price: 120,
        },
      ];

      setStats(mockStats);
      setRecentOrders(mockRecentOrders);
      setIsLoading(false);
    };

    fetchDashboardData();
  }, []);

  // Helper function to get status styling
  const getStatusStyle = (status: RecentOrder['status']) => {
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
  const getStatusLabel = (status: RecentOrder['status']) => {
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

  return (
    <div>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Tableau de bord</h1>
          <p className="text-gray-400">
            Bienvenue, {user?.name} ! Voici un aperçu de votre activité.
          </p>
        </div>
        <div className="mt-4 md:mt-0">
          <Link
            to="/dashboard/services/add"
            className="inline-flex items-center bg-workit-purple text-white px-4 py-2 rounded-md hover:bg-workit-purple-light transition"
          >
            <span className="mr-2">+ Ajouter un service</span>
          </Link>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, index) => (
          <div
            key={index}
            className="bg-gray-900 rounded-lg p-6 border border-gray-800 hover:border-gray-700 transition"
          >
            <div className="flex justify-between items-start">
              <div>
                <p className="text-gray-400 text-sm mb-1">{stat.title}</p>
                <h3 className="text-2xl font-bold text-white">{stat.value}</h3>
                <div className={`mt-2 text-sm ${stat.change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  <span className="flex items-center">
                    {stat.change >= 0 ? (
                      <ArrowUpRight size={16} className="mr-1" />
                    ) : (
                      <ArrowUpRight size={16} className="mr-1 rotate-180" />
                    )}
                    {Math.abs(stat.change)}%
                    <span className="text-gray-500 ml-1">du mois dernier</span>
                  </span>
                </div>
              </div>
              <div className={`p-3 rounded-md ${stat.iconBg}`}>
                {stat.icon}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Orders */}
      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="flex justify-between items-center p-6 border-b border-gray-800">
          <h2 className="text-xl font-semibold text-white">Commandes récentes</h2>
          <Link
            to="/dashboard/orders"
            className="text-sm text-workit-purple hover:underline"
          >
            Voir toutes
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-gray-400 text-sm border-b border-gray-800">
                <th className="py-4 px-6 font-medium">ID</th>
                <th className="py-4 px-6 font-medium">Client</th>
                <th className="py-4 px-6 font-medium">Service</th>
                <th className="py-4 px-6 font-medium">Date</th>
                <th className="py-4 px-6 font-medium">Statut</th>
                <th className="py-4 px-6 font-medium text-right">Montant</th>
              </tr>
            </thead>
            <tbody>
              {recentOrders.map((order) => (
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
                          src={order.clientAvatar}
                          alt={order.clientName}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <span className="text-white">{order.clientName}</span>
                    </div>
                  </td>
                  <td className="py-4 px-6 text-gray-300">{order.service}</td>
                  <td className="py-4 px-6 text-gray-400">
                    {new Date(order.date).toLocaleDateString('fr-FR')}
                  </td>
                  <td className="py-4 px-6">
                    <span className={`inline-flex px-2 py-1 rounded-full text-xs font-medium ${getStatusStyle(order.status)}`}>
                      {getStatusLabel(order.status)}
                    </span>
                  </td>
                  <td className="py-4 px-6 text-right text-white font-medium">
                    {order.price.toFixed(2)} TND
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quick Actions Section */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Actions rapides</h3>
          <div className="space-y-3">
            <Link
              to="/dashboard/services"
              className="flex items-center px-4 py-3 text-gray-300 rounded-md hover:bg-gray-800 transition"
            >
              <FileText size={18} className="mr-3 text-workit-purple" />
              <span>Gérer mes services</span>
            </Link>
            <Link
              to="/dashboard/services/add"
              className="flex items-center px-4 py-3 text-gray-300 rounded-md hover:bg-gray-800 transition"
            >
              <PlusCircle size={18} className="mr-3 text-workit-purple" />
              <span>Ajouter un nouveau service</span>
            </Link>
            <Link
              to="/dashboard/messages"
              className="flex items-center px-4 py-3 text-gray-300 rounded-md hover:bg-gray-800 transition"
            >
              <MessageSquare size={18} className="mr-3 text-workit-purple" />
              <span>Messages</span>
              <span className="ml-auto bg-workit-purple text-white text-xs px-2 py-1 rounded-full">3</span>
            </Link>
            <Link
              to="/profile"
              className="flex items-center px-4 py-3 text-gray-300 rounded-md hover:bg-gray-800 transition"
            >
              <User size={18} className="mr-3 text-workit-purple" />
              <span>Modifier mon profil</span>
            </Link>
          </div>
        </div>

        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 md:col-span-2">
          <h3 className="text-lg font-semibold text-white mb-4">Performance de vos services</h3>
          <div className="space-y-4">
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <div className="text-white font-medium">Développement Web Fullstack</div>
                <div className="text-workit-purple font-medium">150 TND</div>
              </div>
              <div className="flex justify-between text-sm text-gray-400 mb-3">
                <div>10 ventes</div>
                <div>1500 TND au total</div>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-workit-purple h-2 rounded-full" style={{ width: '80%' }}></div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <div className="text-white font-medium">Création de Logo</div>
                <div className="text-workit-purple font-medium">80 TND</div>
              </div>
              <div className="flex justify-between text-sm text-gray-400 mb-3">
                <div>5 ventes</div>
                <div>400 TND au total</div>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-workit-purple h-2 rounded-full" style={{ width: '40%' }}></div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <div className="text-white font-medium">Développement d'API</div>
                <div className="text-workit-purple font-medium">120 TND</div>
              </div>
              <div className="flex justify-between text-sm text-gray-400 mb-3">
                <div>3 ventes</div>
                <div>360 TND au total</div>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-workit-purple h-2 rounded-full" style={{ width: '25%' }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardHome;
