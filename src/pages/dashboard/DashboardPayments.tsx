import { useState } from 'react';
import { usePayment } from '../../context/PaymentContext';
import { useUser } from '../../context/UserContext';
import {
  CreditCard,
  Plus,
  Trash2,
  Check,
  ExternalLink,
  DollarSign,
  Calendar,
  ChevronDown,
  ChevronUp,
  FileText,
} from 'lucide-react';

const DashboardPayments = () => {
  const { user } = useUser();
  const { paymentMethods, transactions, addPaymentMethod, removePaymentMethod, setDefaultPaymentMethod } = usePayment();
  const [isAddingMethod, setIsAddingMethod] = useState(false);
  const [isShowingTransactions, setIsShowingTransactions] = useState(true);
  const [newMethodType, setNewMethodType] = useState<'credit_card' | 'paypal' | 'bank_transfer'>('credit_card');
  const [newMethodName, setNewMethodName] = useState('');
  const [newMethodLast4, setNewMethodLast4] = useState('');
  const [newMethodExpiry, setNewMethodExpiry] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const userTransactions = user ? transactions.filter(t => t.userId === user.id) : [];

  const handleAddPaymentMethod = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    try {
      const method = {
        type: newMethodType,
        name: newMethodName,
        last4: newMethodType === 'credit_card' ? newMethodLast4 : undefined,
        expiryDate: newMethodType === 'credit_card' ? newMethodExpiry : undefined,
        isDefault: paymentMethods.length === 0 ? true : false,
      };

      await addPaymentMethod(method);
      setSuccess('Méthode de paiement ajoutée avec succès');
      resetForm();
    } catch (error) {
      console.error('Error adding payment method:', error);
      setError('Erreur lors de l\'ajout de la méthode de paiement');
    }
  };

  const handleRemovePaymentMethod = async (id: string) => {
    setError(null);
    setSuccess(null);

    try {
      await removePaymentMethod(id);
      setSuccess('Méthode de paiement supprimée avec succès');
    } catch (error) {
      console.error('Error removing payment method:', error);
      setError('Erreur lors de la suppression de la méthode de paiement');
    }
  };

  const handleSetDefaultPaymentMethod = async (id: string) => {
    setError(null);
    setSuccess(null);

    try {
      await setDefaultPaymentMethod(id);
      setSuccess('Méthode de paiement par défaut mise à jour');
    } catch (error) {
      console.error('Error setting default payment method:', error);
      setError('Erreur lors de la mise à jour de la méthode de paiement par défaut');
    }
  };

  const resetForm = () => {
    setIsAddingMethod(false);
    setNewMethodType('credit_card');
    setNewMethodName('');
    setNewMethodLast4('');
    setNewMethodExpiry('');
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('fr-FR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  return (
    <div>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
        <h1 className="text-2xl font-bold text-white mb-2 md:mb-0">Paiements et transactions</h1>
        <button
          onClick={() => setIsAddingMethod(true)}
          className="inline-flex items-center bg-workit-purple text-white px-4 py-2 rounded-md hover:bg-workit-purple-light transition"
          disabled={isAddingMethod}
        >
          <Plus size={18} className="mr-2" />
          Ajouter une méthode de paiement
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 text-red-500 p-4 rounded-md mb-6">
          <p>{error}</p>
        </div>
      )}

      {success && (
        <div className="bg-green-500/10 text-green-500 p-4 rounded-md mb-6">
          <p>{success}</p>
        </div>
      )}

      {isAddingMethod && (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-white">Ajouter une méthode de paiement</h2>
            <button
              onClick={resetForm}
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
                className="w-full p-3 rounded-md bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-workit-purple"
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
                className="w-full p-3 rounded-md bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-workit-purple"
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
                    className="w-full p-3 rounded-md bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-workit-purple"
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
                    className="w-full p-3 rounded-md bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-workit-purple"
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
                onClick={resetForm}
                className="bg-gray-700 text-white px-6 py-2 rounded-md hover:bg-gray-600 transition"
              >
                Annuler
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Payment Methods */}
        <div className="md:col-span-2">
          <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
            <div className="p-6 border-b border-gray-800">
              <h2 className="text-xl font-semibold text-white">Méthodes de paiement</h2>
            </div>

            {paymentMethods.length === 0 ? (
              <div className="p-6 text-center">
                <div className="mx-auto w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mb-4">
                  <CreditCard size={24} className="text-gray-400" />
                </div>
                <h3 className="text-white font-medium mb-1">Aucune méthode de paiement</h3>
                <p className="text-gray-400 mb-4">
                  Ajoutez une méthode de paiement pour commencer à effectuer des transactions.
                </p>
                <button
                  onClick={() => setIsAddingMethod(true)}
                  className="inline-flex items-center bg-workit-purple text-white px-4 py-2 rounded-md hover:bg-workit-purple-light transition"
                >
                  <Plus size={18} className="mr-2" />
                  Ajouter une méthode
                </button>
              </div>
            ) : (
              <div className="divide-y divide-gray-800">
                {paymentMethods.map((method) => (
                  <div key={method.id} className="p-4 flex items-center justify-between">
                    <div className="flex items-center">
                      {method.type === 'credit_card' ? (
                        <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center mr-3">
                          <CreditCard size={20} className="text-workit-purple" />
                        </div>
                      ) : method.type === 'paypal' ? (
                        <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center mr-3">
                          <span className="text-blue-400 font-bold">P</span>
                        </div>
                      ) : (
                        <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center mr-3">
                          <DollarSign size={20} className="text-green-400" />
                        </div>
                      )}
                      <div>
                        <div className="flex items-center">
                          <p className="text-white font-medium">{method.name}</p>
                          {method.isDefault && (
                            <span className="ml-2 px-2 py-0.5 bg-workit-purple/20 text-workit-purple text-xs rounded-full">
                              Par défaut
                            </span>
                          )}
                        </div>
                        {method.type === 'credit_card' && (
                          <p className="text-sm text-gray-400">
                            {`••••${method.last4}`} {method.expiryDate && `- Expire le ${method.expiryDate}`}
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="flex space-x-2">
                      {!method.isDefault && (
                        <button
                          onClick={() => handleSetDefaultPaymentMethod(method.id)}
                          className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-md transition-all"
                          title="Définir par défaut"
                        >
                          <Check size={18} />
                        </button>
                      )}
                      <button
                        onClick={() => handleRemovePaymentMethod(method.id)}
                        className="p-2 text-gray-400 hover:text-red-500 hover:bg-gray-800 rounded-md transition-all"
                        title="Supprimer"
                      >
                        <Trash2 size={18} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Payment Summary */}
        <div>
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
            <h2 className="text-xl font-semibold text-white mb-4">Résumé des paiements</h2>

            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-gray-800 rounded-md">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center mr-3">
                    <DollarSign size={20} className="text-green-500" />
                  </div>
                  <span className="text-white">Total payé</span>
                </div>
                <span className="text-white font-bold">
                  {userTransactions.reduce((sum, t) => sum + t.amount, 0).toFixed(2)} TND
                </span>
              </div>

              <div className="flex justify-between items-center p-3 bg-gray-800 rounded-md">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center mr-3">
                    <FileText size={20} className="text-blue-500" />
                  </div>
                  <span className="text-white">Transactions</span>
                </div>
                <span className="text-white font-bold">
                  {userTransactions.length}
                </span>
              </div>

              <div className="flex justify-between items-center p-3 bg-gray-800 rounded-md">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-full bg-workit-purple/20 flex items-center justify-center mr-3">
                    <CreditCard size={20} className="text-workit-purple" />
                  </div>
                  <span className="text-white">Méthodes de paiement</span>
                </div>
                <span className="text-white font-bold">
                  {paymentMethods.length}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Transactions */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 mb-8">
        <div
          className="flex justify-between items-center p-6 border-b border-gray-800 cursor-pointer"
          onClick={() => setIsShowingTransactions(!isShowingTransactions)}
        >
          <h2 className="text-xl font-semibold text-white">Historique des transactions</h2>
          {isShowingTransactions ? (
            <ChevronUp size={20} className="text-gray-400" />
          ) : (
            <ChevronDown size={20} className="text-gray-400" />
          )}
        </div>

        {isShowingTransactions && (
          <>
            {userTransactions.length === 0 ? (
              <div className="p-6 text-center">
                <div className="mx-auto w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mb-4">
                  <FileText size={24} className="text-gray-400" />
                </div>
                <h3 className="text-white font-medium mb-1">Aucune transaction</h3>
                <p className="text-gray-400">
                  Vos transactions apparaîtront ici une fois que vous aurez effectué des paiements.
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-gray-400 text-sm border-b border-gray-800">
                      <th className="py-4 px-6 font-medium">ID</th>
                      <th className="py-4 px-6 font-medium">Service</th>
                      <th className="py-4 px-6 font-medium">Date</th>
                      <th className="py-4 px-6 font-medium">Statut</th>
                      <th className="py-4 px-6 font-medium text-right">Montant</th>
                    </tr>
                  </thead>
                  <tbody>
                    {userTransactions.map((transaction) => (
                      <tr key={transaction.id} className="border-b border-gray-800 hover:bg-gray-800/50 transition">
                        <td className="py-4 px-6">
                          <span className="text-workit-purple">
                            {transaction.id}
                          </span>
                        </td>
                        <td className="py-4 px-6">
                          <div className="text-white">{transaction.serviceName}</div>
                          <div className="text-gray-400 text-sm">{transaction.orderId}</div>
                        </td>
                        <td className="py-4 px-6 text-gray-400">
                          {formatDate(transaction.timestamp)}
                        </td>
                        <td className="py-4 px-6">
                          <span className={`inline-flex px-2 py-1 rounded-full text-xs font-medium ${
                            transaction.status === 'completed'
                              ? 'bg-green-500/10 text-green-500'
                              : transaction.status === 'failed'
                              ? 'bg-red-500/10 text-red-500'
                              : transaction.status === 'refunded'
                              ? 'bg-orange-500/10 text-orange-500'
                              : 'bg-yellow-500/10 text-yellow-500'
                          }`}>
                            {transaction.status === 'completed'
                              ? 'Terminé'
                              : transaction.status === 'failed'
                              ? 'Échoué'
                              : transaction.status === 'refunded'
                              ? 'Remboursé'
                              : 'En attente'
                            }
                          </span>
                        </td>
                        <td className="py-4 px-6 text-right text-white font-medium">
                          {transaction.amount.toFixed(2)} {transaction.currency}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default DashboardPayments;
