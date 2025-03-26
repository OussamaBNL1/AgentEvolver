import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useUser } from './UserContext';

export interface PaymentMethod {
  id: string;
  type: 'credit_card' | 'paypal' | 'bank_transfer';
  last4?: string;
  expiryDate?: string;
  name?: string;
  isDefault: boolean;
}

export interface Transaction {
  id: string;
  userId: string;
  serviceId: string;
  serviceName: string;
  amount: number;
  currency: string;
  status: 'pending' | 'completed' | 'failed' | 'refunded';
  paymentMethodId: string;
  timestamp: string;
  orderId?: string;
}

interface PaymentContextType {
  paymentMethods: PaymentMethod[];
  transactions: Transaction[];
  addPaymentMethod: (method: Omit<PaymentMethod, 'id'>) => Promise<PaymentMethod>;
  removePaymentMethod: (id: string) => Promise<void>;
  setDefaultPaymentMethod: (id: string) => Promise<void>;
  processPayment: (
    serviceId: string,
    serviceName: string,
    amount: number,
    paymentMethodId?: string
  ) => Promise<Transaction>;
  getTransactionsByUser: (userId: string) => Transaction[];
}

const PaymentContext = createContext<PaymentContextType | undefined>(undefined);

export const usePayment = () => {
  const context = useContext(PaymentContext);
  if (context === undefined) {
    throw new Error('usePayment must be used within a PaymentProvider');
  }
  return context;
};

export const PaymentProvider = ({ children }: { children: ReactNode }) => {
  const { user } = useUser();
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [transactions, setTransactions] = useState<Transaction[]>([]);

  useEffect(() => {
    // Load payment methods and transactions from localStorage
    const storedPaymentMethods = localStorage.getItem('workit_payment_methods');
    const storedTransactions = localStorage.getItem('workit_transactions');

    if (storedPaymentMethods) {
      try {
        const parsedMethods = JSON.parse(storedPaymentMethods);
        setPaymentMethods(parsedMethods);
      } catch (error) {
        console.error('Error parsing stored payment methods:', error);
        localStorage.removeItem('workit_payment_methods');
      }
    } else {
      // Initialize with mock data for demo purposes
      initializeMockData();
    }

    if (storedTransactions) {
      try {
        const parsedTransactions = JSON.parse(storedTransactions);
        setTransactions(parsedTransactions);
      } catch (error) {
        console.error('Error parsing stored transactions:', error);
        localStorage.removeItem('workit_transactions');
      }
    }
  }, []);

  const initializeMockData = () => {
    if (!user) return;

    // Mock payment methods
    const mockPaymentMethods: PaymentMethod[] = [
      {
        id: 'pm_1',
        type: 'credit_card',
        last4: '4242',
        expiryDate: '12/25',
        name: 'Visa',
        isDefault: true,
      },
      {
        id: 'pm_2',
        type: 'paypal',
        name: 'My PayPal Account',
        isDefault: false,
      },
    ];

    // Mock transactions
    const mockTransactions: Transaction[] = [
      {
        id: 'tx_1',
        userId: user.id,
        serviceId: 'svc_1',
        serviceName: 'Développement Web Fullstack',
        amount: 150,
        currency: 'TND',
        status: 'completed',
        paymentMethodId: 'pm_1',
        timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
        orderId: 'ORD-001',
      },
      {
        id: 'tx_2',
        userId: user.id,
        serviceId: 'svc_2',
        serviceName: 'Création de Logo',
        amount: 80,
        currency: 'TND',
        status: 'completed',
        paymentMethodId: 'pm_1',
        timestamp: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
        orderId: 'ORD-002',
      },
    ];

    setPaymentMethods(mockPaymentMethods);
    setTransactions(mockTransactions);

    localStorage.setItem('workit_payment_methods', JSON.stringify(mockPaymentMethods));
    localStorage.setItem('workit_transactions', JSON.stringify(mockTransactions));
  };

  const saveToLocalStorage = () => {
    localStorage.setItem('workit_payment_methods', JSON.stringify(paymentMethods));
    localStorage.setItem('workit_transactions', JSON.stringify(transactions));
  };

  const addPaymentMethod = async (method: Omit<PaymentMethod, 'id'>): Promise<PaymentMethod> => {
    const newMethod: PaymentMethod = {
      ...method,
      id: 'pm_' + Math.random().toString(36).substring(2, 9),
    };

    // If this is the first payment method or isDefault is true, make it the default
    if (paymentMethods.length === 0 || method.isDefault) {
      // Set all other methods to non-default
      const updatedMethods = paymentMethods.map(m => ({ ...m, isDefault: false }));
      const newMethods = [...updatedMethods, newMethod];
      setPaymentMethods(newMethods);
    } else {
      setPaymentMethods([...paymentMethods, newMethod]);
    }

    saveToLocalStorage();
    return newMethod;
  };

  const removePaymentMethod = async (id: string) => {
    const methodToRemove = paymentMethods.find(m => m.id === id);
    if (!methodToRemove) throw new Error('Payment method not found');

    // Don't allow removing the default payment method if it's the only one
    if (methodToRemove.isDefault && paymentMethods.length > 1) {
      // Set a new default
      const newDefaultIndex = paymentMethods.findIndex(m => m.id !== id);
      const updatedMethods = paymentMethods.map((m, index) => ({
        ...m,
        isDefault: index === newDefaultIndex,
      }));

      setPaymentMethods(updatedMethods.filter(m => m.id !== id));
    } else if (paymentMethods.length > 1) {
      setPaymentMethods(paymentMethods.filter(m => m.id !== id));
    } else {
      throw new Error('Cannot remove the only payment method');
    }

    saveToLocalStorage();
  };

  const setDefaultPaymentMethod = async (id: string) => {
    const methodExists = paymentMethods.some(m => m.id === id);
    if (!methodExists) throw new Error('Payment method not found');

    const updatedMethods = paymentMethods.map(method => ({
      ...method,
      isDefault: method.id === id,
    }));

    setPaymentMethods(updatedMethods);
    saveToLocalStorage();
  };

  const processPayment = async (
    serviceId: string,
    serviceName: string,
    amount: number,
    paymentMethodId?: string
  ): Promise<Transaction> => {
    if (!user) throw new Error('User not authenticated');

    // Get the payment method to use (either the specified one or the default)
    let methodId = paymentMethodId;
    if (!methodId) {
      const defaultMethod = paymentMethods.find(m => m.isDefault);
      if (!defaultMethod) throw new Error('No default payment method found');
      methodId = defaultMethod.id;
    }

    // In a real app, this would call a payment API
    // For demo purposes, we'll simulate a successful payment
    await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API delay

    const transaction: Transaction = {
      id: 'tx_' + Math.random().toString(36).substring(2, 9),
      userId: user.id,
      serviceId,
      serviceName,
      amount,
      currency: 'TND',
      status: 'completed', // For demo, all payments succeed
      paymentMethodId: methodId,
      timestamp: new Date().toISOString(),
      orderId: 'ORD-' + Math.floor(1000 + Math.random() * 9000),
    };

    setTransactions([...transactions, transaction]);
    saveToLocalStorage();

    return transaction;
  };

  const getTransactionsByUser = (userId: string): Transaction[] => {
    return transactions.filter(t => t.userId === userId);
  };

  return (
    <PaymentContext.Provider
      value={{
        paymentMethods,
        transactions,
        addPaymentMethod,
        removePaymentMethod,
        setDefaultPaymentMethod,
        processPayment,
        getTransactionsByUser,
      }}
    >
      {children}
    </PaymentContext.Provider>
  );
};
