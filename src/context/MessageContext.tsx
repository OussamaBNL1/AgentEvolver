import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useUser } from './UserContext';

export interface Message {
  id: string;
  senderId: string;
  receiverId: string;
  conversationId: string;
  content: string;
  timestamp: string;
  isRead: boolean;
}

export interface Conversation {
  id: string;
  participants: string[]; // User IDs
  lastMessage?: Message;
  title?: string; // Title for the conversation (e.g., service name)
}

interface MessageContextType {
  conversations: Conversation[];
  messages: Record<string, Message[]>; // Key is conversationId
  unreadCount: number;
  sendMessage: (conversationId: string, content: string) => Promise<void>;
  createConversation: (participantId: string, title?: string) => Promise<string>;
  markAsRead: (conversationId: string) => Promise<void>;
  getConversationMessages: (conversationId: string) => Message[];
  getConversation: (conversationId: string) => Conversation | undefined;
}

const MessageContext = createContext<MessageContextType | undefined>(undefined);

export const useMessages = () => {
  const context = useContext(MessageContext);
  if (context === undefined) {
    throw new Error('useMessages must be used within a MessageProvider');
  }
  return context;
};

export const MessageProvider = ({ children }: { children: ReactNode }) => {
  const { user } = useUser();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<Record<string, Message[]>>({});
  const [unreadCount, setUnreadCount] = useState<number>(0);

  useEffect(() => {
    // Load conversations and messages from localStorage
    const storedConversations = localStorage.getItem('workit_conversations');
    const storedMessages = localStorage.getItem('workit_messages');

    if (storedConversations) {
      try {
        const parsedConversations = JSON.parse(storedConversations);
        setConversations(parsedConversations);
      } catch (error) {
        console.error('Error parsing stored conversations:', error);
        localStorage.removeItem('workit_conversations');
      }
    } else {
      // Initialize with mock data for demo purposes
      initializeMockData();
    }

    if (storedMessages) {
      try {
        const parsedMessages = JSON.parse(storedMessages);
        setMessages(parsedMessages);
      } catch (error) {
        console.error('Error parsing stored messages:', error);
        localStorage.removeItem('workit_messages');
      }
    }
  }, []);

  useEffect(() => {
    // Calculate unread count when messages change
    if (user) {
      let count = 0;
      Object.values(messages).forEach(conversationMessages => {
        count += conversationMessages.filter(
          msg => msg.receiverId === user.id && !msg.isRead
        ).length;
      });
      setUnreadCount(count);
    }
  }, [messages, user]);

  const initializeMockData = () => {
    if (!user) return;

    // Mock conversation and messages
    const mockConversations: Conversation[] = [
      {
        id: 'conv_1',
        participants: [user.id, 'client_1'],
        title: 'Développement Web Fullstack',
      },
      {
        id: 'conv_2',
        participants: [user.id, 'client_2'],
        title: 'Création de Logo',
      },
      {
        id: 'conv_3',
        participants: [user.id, 'client_3'],
        title: 'Développement d\'API',
      }
    ];

    const mockMessages: Record<string, Message[]> = {
      'conv_1': [
        {
          id: 'msg_1',
          senderId: 'client_1',
          receiverId: user.id,
          conversationId: 'conv_1',
          content: 'Bonjour, j\'aimerais discuter du projet de développement web.',
          timestamp: new Date(Date.now() - 3600000 * 24).toISOString(),
          isRead: true,
        },
        {
          id: 'msg_2',
          senderId: user.id,
          receiverId: 'client_1',
          conversationId: 'conv_1',
          content: 'Bonjour ! Bien sûr, je suis disponible pour en discuter. Quels sont vos besoins ?',
          timestamp: new Date(Date.now() - 3600000 * 23).toISOString(),
          isRead: true,
        },
        {
          id: 'msg_3',
          senderId: 'client_1',
          receiverId: user.id,
          conversationId: 'conv_1',
          content: 'J\'ai besoin d\'un site e-commerce avec une intégration de paiement.',
          timestamp: new Date(Date.now() - 3600000 * 2).toISOString(),
          isRead: false,
        },
      ],
      'conv_2': [
        {
          id: 'msg_4',
          senderId: 'client_2',
          receiverId: user.id,
          conversationId: 'conv_2',
          content: 'Pouvez-vous me créer un logo pour ma nouvelle startup ?',
          timestamp: new Date(Date.now() - 3600000 * 48).toISOString(),
          isRead: true,
        },
        {
          id: 'msg_5',
          senderId: user.id,
          receiverId: 'client_2',
          conversationId: 'conv_2',
          content: 'Bien sûr ! Pouvez-vous me donner plus de détails sur votre entreprise ?',
          timestamp: new Date(Date.now() - 3600000 * 47).toISOString(),
          isRead: true,
        },
      ],
      'conv_3': [
        {
          id: 'msg_6',
          senderId: 'client_3',
          receiverId: user.id,
          conversationId: 'conv_3',
          content: 'Bonjour, j\'ai besoin d\'une API pour mon application mobile.',
          timestamp: new Date(Date.now() - 3600000 * 5).toISOString(),
          isRead: false,
        },
      ],
    };

    setConversations(mockConversations);
    setMessages(mockMessages);

    localStorage.setItem('workit_conversations', JSON.stringify(mockConversations));
    localStorage.setItem('workit_messages', JSON.stringify(mockMessages));
  };

  const saveToLocalStorage = () => {
    localStorage.setItem('workit_conversations', JSON.stringify(conversations));
    localStorage.setItem('workit_messages', JSON.stringify(messages));
  };

  const sendMessage = async (conversationId: string, content: string) => {
    if (!user) return;

    const conversation = conversations.find(c => c.id === conversationId);
    if (!conversation) throw new Error('Conversation not found');

    const receiverId = conversation.participants.find(id => id !== user.id) || '';

    const newMessage: Message = {
      id: 'msg_' + Math.random().toString(36).substring(2, 9),
      senderId: user.id,
      receiverId,
      conversationId,
      content,
      timestamp: new Date().toISOString(),
      isRead: false,
    };

    const updatedMessages = {
      ...messages,
      [conversationId]: [...(messages[conversationId] || []), newMessage],
    };

    const updatedConversations = conversations.map(c =>
      c.id === conversationId
        ? { ...c, lastMessage: newMessage }
        : c
    );

    setMessages(updatedMessages);
    setConversations(updatedConversations);
    saveToLocalStorage();
  };

  const createConversation = async (participantId: string, title?: string) => {
    if (!user) throw new Error('User not authenticated');

    // Check if conversation already exists
    const existingConversation = conversations.find(c =>
      c.participants.includes(user.id) &&
      c.participants.includes(participantId)
    );

    if (existingConversation) return existingConversation.id;

    const newConversation: Conversation = {
      id: 'conv_' + Math.random().toString(36).substring(2, 9),
      participants: [user.id, participantId],
      title,
    };

    const updatedConversations = [...conversations, newConversation];
    setConversations(updatedConversations);
    saveToLocalStorage();

    return newConversation.id;
  };

  const markAsRead = async (conversationId: string) => {
    if (!user) return;

    const conversationMessages = messages[conversationId];
    if (!conversationMessages) return;

    const updatedConversationMessages = conversationMessages.map(msg =>
      msg.receiverId === user.id && !msg.isRead
        ? { ...msg, isRead: true }
        : msg
    );

    const updatedMessages = {
      ...messages,
      [conversationId]: updatedConversationMessages,
    };

    setMessages(updatedMessages);
    saveToLocalStorage();
  };

  const getConversationMessages = (conversationId: string) => {
    return messages[conversationId] || [];
  };

  const getConversation = (conversationId: string) => {
    return conversations.find(c => c.id === conversationId);
  };

  return (
    <MessageContext.Provider
      value={{
        conversations,
        messages,
        unreadCount,
        sendMessage,
        createConversation,
        markAsRead,
        getConversationMessages,
        getConversation,
      }}
    >
      {children}
    </MessageContext.Provider>
  );
};
