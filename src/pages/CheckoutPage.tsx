import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { usePayment } from '../context/PaymentContext';
import {
  CreditCard,
  ShoppingCart,
  Shield,
  ChevronRight,
  CheckCircle,
  ArrowLeft,
  Plus,
} from 'lucide-react';

interface Service {
  id: string;
  title: string;
  description: string;
  price: number;
  category: string;
  image?: string;
  userId: string;
}

const CheckoutPage = () => {
  const { serviceId } = useParams<{ serviceId: string }>();
  const navigate = useNavigate();
  const { user, isAuthenticated } = useUser();
  const { paymentMethods, addPaymentMethod, processPayment } = usePayment();
  const [service, setService] = useState<Service | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPaymentMethodId, setSelectedPaymentMethodId] = useState<string | null>(null);
  const [isAddingPaymentMethod, setIsAddingPaymentMethod] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [transactionId, setTransactionId] = useState<string | null>(null);
  const [orderId, setOrderId] = useState<string | null>(null);

  // Form state for adding new payment method
  const [newMethodType, setNewMethodType] = useState<'credit_card' | 'paypal' | 'bank_transfer'>('credit_card');
  const [newMethodName, setNewMethodName] = useState('');
  const [newMethodLast4, setNewMethodLast4] = useState('');
  const [newMethodExpiry, setNewMethodExpiry] = useState('');

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login', { state: { from: `/checkout/${serviceId}` } });
      return;
    }

    const fetchService = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // In a real app, this would be an API call
        const storedServices = localStorage.getItem('workit_services');

        if (storedServices) {
          const services: Service[] = JSON.parse(storedServices);
          const service = services.find(s => s.id === serviceId);

          if (service) {
            setService(service);
            // Set default payment method if available
            const defaultMethod = paymentMethods.find(m => m.isDefault);
            if (defaultMethod) {
              setSelectedPaymentMethodId(defaultMethod.id);
            }
          } else {
            setError('Service non trouvé');
          }
        } else {
          // For demo purposes, create a mock service
          const mockService: Service = {
            id: serviceId || 'svc_demo',
            title: 'Développement Web Fullstack',
            description: 'Service de développement web complet avec frontend et backend',
            price: 150,
            category: 'web',
            userId: 'seller_1',
          };
          setService(mockService);
          // Set default payment method if available
          const defaultMethod = paymentMethods.find(m => m.isDefault);
          if (defaultMethod) {
            setSelectedPaymentMethodId(defaultMethod.id);
          }
        }
      } catch (err) {
        console.error('Error fetching service:', err);
        setError('Erreur lors du chargement du service');
      } finally {
        setIsLoading(false);
      }
    };

    fetchService();
  }, [serviceId, navigate, isAuthenticated, paymentMethods]);

  const handleAddPaymentMethod = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    try {
      const method = {
        type: newMethodType,
        name: newMethodName,
        last4: newMethodType === 'credit_card' ? newMethodLast4 : undefined,
        expiryDate: newMethodType === 'credit_card' ? newMethodExpiry : undefined,
        isDefault: paymentMethods.length === 0 ? true : false,
      };

      const newMethod = await addPaymentMethod(method);
      setSelectedPaymentMethodId(newMethod.id);
      setIsAddingPaymentMethod(false);
      resetForm();
    } catch (error) {
      console.error('Error adding payment method:', error);
      setError('Erreur lors de l\'ajout de la méthode de paiement');
    }
  };

  const handleCheckout = async () => {
    if (!service || !selectedPaymentMethodId) return;

    setIsProcessing(true);
    setError(null);

    try {
      const transaction = await processPayment(
        service.id,
        service.title,
        service.price,
        selectedPaymentMethodId
      );

      // In a real app, this would create an order in the database
      setTransactionId(transaction.id);
      setOrderId(transaction.orderId || 'ORD-0000');
      setIsComplete(true);
    } catch (error) {
      console.error('Error processing payment:', error);
      setError('Erreur lors du traitement du paiement');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetForm = () => {
    setNewMethodType('credit_card');
    setNewMethodName('');
    setNewMethodLast4('');
    setNewMethodExpiry('');
  };

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-16">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-workit-purple"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-16">
        <div className="bg-red-500/10 text-red-500 p-4 rounded-md mb-6">
          <p>{error}</p>
        </div>
        <Link to="/services" className="text-workit-purple hover:underline">
          Retour aux services
        </Link>
      </div>
    );
  }

  if (!service) {
    return (
      <div className="container mx-auto px-4 py-16">
        <div className="bg-yellow-500/10 text-yellow-500 p-4 rounded-md mb-6">
          <p>Service non trouvé</p>
        </div>
        <Link to="/services" className="text-workit-purple hover:underline">
          Retour aux services
        </Link>
      </div>
    );
  }

  if (isComplete) {
    return (
      <div className="container mx-auto px-4 py-16 max-w-4xl">
        <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-800">
          <div className="p-6 border-b border-gray-800 text-center">
            <div className="mx-auto w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mb-4">
              <CheckCircle size={32} className="text-green-500" />
            </div>
            <h1 className="text-2xl font-bold text-white mb-2">Paiement effectué</h1>
            <p className="text-gray-400">
              Votre commande a été traitée avec succès.
            </p>
          </div>

          <div className="p-6">
            <div className="bg-gray-800 rounded-lg p-6 mb-6">
              <h2 className="text-xl font-semibold text-white mb-4">Récapitulatif de la commande</h2>

              <div className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-gray-400">ID de transaction :</span>
                  <span className="text-white font-medium">{transactionId}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-gray-400">Numéro de commande :</span>
                  <span className="text-white font-medium">{orderId}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-gray-400">Service :</span>
                  <span className="text-white font-medium">{service.title}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-gray-400">Prix :</span>
                  <span className="text-white font-medium">{service.price.toFixed(2)} TND</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-gray-400">Date :</span>
                  <span className="text-white font-medium">
                    {new Date().toLocaleDateString('fr-FR')}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3">
              <Link
                to="/dashboard/orders"
                className="flex-1 bg-workit-purple text-white px-6 py-3 rounded-md hover:bg-workit-purple-light transition text-center"
              >
                Voir mes commandes
              </Link>
              <Link
                to="/services"
                className="flex-1 bg-gray-700 text-white px-6 py-3 rounded-md hover:bg-gray-600 transition text-center"
              >
                Parcourir d'autres services
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-16 max-w-5xl">
      <div className="flex items-center space-x-2 text-gray-400 mb-6">
        <Link to="/services" className="hover:text-white transition">
          Services
        </Link>
        <ChevronRight size={16} />
        <Link to={`/service/${service.id}`} className="hover:text-white transition">
          {service.title}
        </Link>
        <ChevronRight size={16} />
        <span className="text-white">Paiement</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Checkout Form */}
        <div className="lg:col-span-2">
          <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-800 mb-6">
            <div className="p-6 border-b border-gray-800">
              <h1 className="text-2xl font-bold text-white">Paiement</h1>
            </div>

            <div className="p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Méthode de paiement</h2>

              {paymentMethods.length === 0 ? (
                <div className="bg-gray-800 rounded-lg p-6 mb-6">
                  <p className="text-gray-300 mb-4">
                    Vous n'avez pas encore de méthode de paiement. Veuillez en ajouter une pour continuer.
                  </p>
                  <button
                    onClick={() => setIsAddingPaymentMethod(true)}
                    className="bg-workit-purple text-white px-4 py-2 rounded-md hover:bg-workit-purple-light transition flex items-center"
                  >
                    <Plus size={18} className="mr-2" />
                    Ajouter une méthode de paiement
                  </button>
                </div>
              ) : (
                <div className="space-y-4 mb-6">
                  {paymentMethods.map((method) => (
                    <div
                      key={method.id}
                      className={`p-4 border rounded-lg flex items-center cursor-pointer transition ${
                        selectedPaymentMethodId === method.id
                          ? 'border-workit-purple bg-workit-purple/10'
                          : 'border-gray-700 bg-gray-800 hover:border-gray-600'
                      }`}
                      onClick={() => setSelectedPaymentMethodId(method.id)}
                    >
                      <div className="w-6 h-6 rounded-full border-2 flex items-center justify-center mr-3 transition-colors duration-200 ease-in-out border-workit-purple">
                        {selectedPaymentMethodId === method.id && (
                          <div className="w-3 h-3 rounded-full bg-workit-purple"></div>
                        )}
                      </div>

                      {method.type === 'credit_card' ? (
                        <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center mr-3">
                          <CreditCard size={20} className="text-workit-purple" />
                        </div>
                      ) : method.type === 'paypal' ? (
                        <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center mr-3">
                          <span className="text-blue-400 font-bold">P</span>
                        </div>
                      ) : (
                        <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center mr-3">
                          <span className="text-green-500 font-bold">$</span>
                        </div>
                      )}

                      <div>
                        <p className="text-white font-medium">{method.name}</p>
                        {method.type === 'credit_card' && method.last4 && (
                          <p className="text-gray-400 text-sm">
                            •••• {method.last4} {method.expiryDate && `(${method.expiryDate})`}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}

                  <button
                    onClick={() => setIsAddingPaymentMethod(true)}
                    className="w-full p-4 border border-dashed border-gray-700 rounded-lg bg-gray-800 hover:bg-gray-700 transition flex items-center justify-center"
                  >
                    <Plus size={18} className="mr-2 text-workit-purple" />
                    <span className="text-white">Ajouter une nouvelle méthode de paiement</span>
                  </button>
                </div>
              )}

              {isAddingPaymentMethod && (
                <div className="bg-gray-800 rounded-lg p-6 mb-6">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-medium text-white">Ajouter une méthode de paiement</h3>
                    <button
                      onClick={() => setIsAddingPaymentMethod(false)}
                      className="text-gray-400 hover:text-white"
                    >
                      &times;
                    </button>
                  </div>

                  <form onSubmit={handleAddPaymentMethod}>
                    <div className="mb-4">
                      <label htmlFor="paymentType" className="block text-white mb-2">
                        Type de paiement
                      </label>
                      <select
                        id="paymentType"
                        className="w-full p-3 rounded-md bg-gray-700 border border-gray-600 text-white focus:outline-none focus:border-workit-purple"
                        value={newMethodType}
                        onChange={(e) => setNewMethodType(e.target.value as any)}
                      >
                        <option value="credit_card">Carte de crédit</option>
                        <option value="paypal">PayPal</option>
                        <option value="bank_transfer">Virement bancaire</option>
                      </select>
                    </div>

                    <div className="mb-4">
                      <label htmlFor="methodName" className="block text-white mb-2">
                        Nom de la méthode
                      </label>
                      <input
                        type="text"
                        id="methodName"
                        className="w-full p-3 rounded-md bg-gray-700 border border-gray-600 text-white focus:outline-none focus:border-workit-purple"
                        placeholder={newMethodType === 'credit_card' ? 'Visa, Mastercard, etc.' : 'Nom du compte'}
                        value={newMethodName}
                        onChange={(e) => setNewMethodName(e.target.value)}
                        required
                      />
                    </div>

                    {newMethodType === 'credit_card' && (
                      <>
                        <div className="mb-4">
                          <label htmlFor="last4" className="block text-white mb-2">
                            4 derniers chiffres
                          </label>
                          <input
                            type="text"
                            id="last4"
                            className="w-full p-3 rounded-md bg-gray-700 border border-gray-600 text-white focus:outline-none focus:border-workit-purple"
                            placeholder="1234"
                            value={newMethodLast4}
                            onChange={(e) => setNewMethodLast4(e.target.value)}
                            maxLength={4}
                            pattern="[0-9]{4}"
                            required
                          />
                        </div>

                        <div className="mb-4">
                          <label htmlFor="expiry" className="block text-white mb-2">
                            Date d'expiration
                          </label>
                          <input
                            type="text"
                            id="expiry"
                            className="w-full p-3 rounded-md bg-gray-700 border border-gray-600 text-white focus:outline-none focus:border-workit-purple"
                            placeholder="MM/YY"
                            value={newMethodExpiry}
                            onChange={(e) => setNewMethodExpiry(e.target.value)}
                            maxLength={5}
                            pattern="[0-9]{2}\/[0-9]{2}"
                            required
                          />
                        </div>
                      </>
                    )}

                    <div className="flex space-x-4">
                      <button
                        type="submit"
                        className="bg-workit-purple text-white px-6 py-2 rounded-md hover:bg-workit-purple-light transition"
                      >
                        Ajouter
                      </button>
                      <button
                        type="button"
                        onClick={() => setIsAddingPaymentMethod(false)}
                        className="bg-gray-700 text-white px-6 py-2 rounded-md hover:bg-gray-600 transition"
                      >
                        Annuler
                      </button>
                    </div>
                  </form>
                </div>
              )}

              <div className="mt-6">
                <button
                  onClick={handleCheckout}
                  disabled={!selectedPaymentMethodId || isProcessing}
                  className="w-full bg-workit-purple text-white py-3 rounded-md hover:bg-workit-purple-light transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isProcessing ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                      Traitement en cours...
                    </>
                  ) : (
                    <>
                      <ShoppingCart size={20} className="mr-2" />
                      Payer {service.price.toFixed(2)} TND
                    </>
                  )}
                </button>
              </div>

              <div className="mt-4 text-center">
                <p className="text-gray-400 text-sm flex items-center justify-center">
                  <Shield size={16} className="mr-2" />
                  Paiement sécurisé
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Order Summary */}
        <div>
          <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-800 sticky top-6">
            <div className="p-6 border-b border-gray-800">
              <h2 className="text-xl font-semibold text-white">Résumé de la commande</h2>
            </div>

            <div className="p-6">
              <div className="flex items-center mb-4">
                <div className="w-16 h-16 bg-gray-800 rounded flex items-center justify-center mr-3">
                  {service.image ? (
                    <img src={service.image} alt={service.title} className="w-full h-full object-cover rounded" />
                  ) : (
                    <div className="text-gray-500 font-semibold">{service.title.charAt(0)}</div>
                  )}
                </div>
                <div>
                  <h3 className="text-white font-medium">{service.title}</h3>
                  <p className="text-gray-400 text-sm">{service.category}</p>
                </div>
              </div>

              <div className="border-t border-gray-800 pt-4 mt-4">
                <div className="flex justify-between mb-2">
                  <span className="text-gray-400">Prix</span>
                  <span className="text-white">{service.price.toFixed(2)} TND</span>
                </div>
                <div className="flex justify-between mb-2">
                  <span className="text-gray-400">Taxes</span>
                  <span className="text-white">0.00 TND</span>
                </div>
                <div className="border-t border-gray-800 pt-4 mt-4">
                  <div className="flex justify-between font-bold">
                    <span className="text-white">Total</span>
                    <span className="text-workit-purple">{service.price.toFixed(2)} TND</span>
                  </div>
                </div>
              </div>

              <div className="mt-6">
                <Link
                  to={`/service/${service.id}`}
                  className="flex items-center text-workit-purple hover:underline"
                >
                  <ArrowLeft size={16} className="mr-1" />
                  Retour aux détails du service
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CheckoutPage;
