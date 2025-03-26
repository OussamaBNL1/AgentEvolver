import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useMessages } from '../../context/MessageContext';
import { useUser } from '../../context/UserContext';
import { Search, MessageSquare, ChevronRight } from 'lucide-react';

const DashboardMessages = () => {
  const { user } = useUser();
  const { conversations, messages, markAsRead } = useMessages();
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  // Mock client data
  const clientData: Record<string, { name: string; avatar?: string }> = {
    client_1: {
      name: 'Jean Dupont',
      avatar: 'https://randomuser.me/api/portraits/men/1.jpg',
    },
    client_2: {
      name: 'Sophie Martin',
      avatar: 'https://randomuser.me/api/portraits/women/2.jpg',
    },
    client_3: {
      name: 'Marc Lefevre',
      avatar: 'https://randomuser.me/api/portraits/men/3.jpg',
    },
  };

  useEffect(() => {
    // Simulating loading data
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  const getClientInfo = (participantIds: string[]) => {
    if (!user) return { name: 'Unknown', avatar: undefined };

    const clientId = participantIds.find(id => id !== user.id);
    if (!clientId) return { name: 'Unknown', avatar: undefined };

    return clientData[clientId] || { name: 'Unknown', avatar: undefined };
  };

  const getLastMessageTimestamp = (conversationId: string) => {
    const conversationMessages = messages[conversationId];
    if (!conversationMessages || conversationMessages.length === 0) return '';

    const lastMessage = conversationMessages[conversationMessages.length - 1];
    const date = new Date(lastMessage.timestamp);

    // If it's today, show the time, otherwise show the date
    const now = new Date();
    if (
      date.getDate() === now.getDate() &&
      date.getMonth() === now.getMonth() &&
      date.getFullYear() === now.getFullYear()
    ) {
      return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
    } else {
      return date.toLocaleDateString('fr-FR');
    }
  };

  const getLastMessagePreview = (conversationId: string) => {
    const conversationMessages = messages[conversationId];
    if (!conversationMessages || conversationMessages.length === 0) return 'Aucun message';

    const lastMessage = conversationMessages[conversationMessages.length - 1];
    return lastMessage.content.length > 50
      ? `${lastMessage.content.substring(0, 50)}...`
      : lastMessage.content;
  };

  const getUnreadCountForConversation = (conversationId: string) => {
    if (!user) return 0;

    const conversationMessages = messages[conversationId];
    if (!conversationMessages) return 0;

    return conversationMessages.filter(
      msg => msg.receiverId === user.id && !msg.isRead
    ).length;
  };

  const filteredConversations = conversations.filter(conversation => {
    if (!searchTerm.trim()) return true;

    const client = getClientInfo(conversation.participants);
    return (
      client.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (conversation.title?.toLowerCase().includes(searchTerm.toLowerCase()))
    );
  });

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
        <h1 className="text-2xl font-bold text-white mb-2 md:mb-0">Messages</h1>
        <div className="w-full md:w-64">
          <div className="relative">
            <input
              type="text"
              placeholder="Rechercher..."
              className="w-full px-4 py-2 pl-10 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:border-workit-purple"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <Search
              size={18}
              className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
            />
          </div>
        </div>
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800">
        {filteredConversations.length === 0 ? (
          <div className="p-6 text-center">
            <div className="mx-auto w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mb-4">
              <MessageSquare size={24} className="text-gray-400" />
            </div>
            <h3 className="text-white font-medium mb-1">Aucune conversation</h3>
            <p className="text-gray-400">
              {searchTerm
                ? "Aucune conversation trouvée pour cette recherche."
                : "Vous n'avez pas encore de messages. Commencez par contacter un client."}
            </p>
          </div>
        ) : (
          <div className="divide-y divide-gray-800">
            {filteredConversations.map((conversation) => {
              const client = getClientInfo(conversation.participants);
              const unreadCount = getUnreadCountForConversation(conversation.id);

              return (
                <Link
                  key={conversation.id}
                  to={`/dashboard/messages/${conversation.id}`}
                  className="flex items-center p-4 hover:bg-gray-800 transition"
                >
                  <div className="flex-shrink-0 relative">
                    <div className="w-12 h-12 rounded-full overflow-hidden">
                      {client.avatar ? (
                        <img
                          src={client.avatar}
                          alt={client.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full bg-gray-700 flex items-center justify-center text-white">
                          {client.name.charAt(0).toUpperCase()}
                        </div>
                      )}
                    </div>
                    {unreadCount > 0 && (
                      <div className="absolute top-0 right-0 w-4 h-4 bg-workit-purple rounded-full flex items-center justify-center text-white text-xs">
                        {unreadCount}
                      </div>
                    )}
                  </div>

                  <div className="ml-4 flex-1 min-w-0">
                    <div className="flex justify-between items-center mb-1">
                      <h4 className="text-white font-medium truncate">{client.name}</h4>
                      <span className="text-xs text-gray-400">
                        {getLastMessageTimestamp(conversation.id)}
                      </span>
                    </div>

                    <div className="flex justify-between items-center">
                      <p className="text-gray-400 text-sm truncate">
                        {getLastMessagePreview(conversation.id)}
                      </p>
                      <ChevronRight size={16} className="text-gray-500 ml-2 flex-shrink-0" />
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default DashboardMessages;
