import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useMessages } from '../../context/MessageContext';
import { useUser } from '../../context/UserContext';
import { ArrowLeft, Send, Clock, Check, ChevronLeft, Calendar } from 'lucide-react';

const DashboardMessageDetails = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user } = useUser();
  const { getConversationMessages, getConversation, sendMessage, markAsRead } = useMessages();
  const [newMessage, setNewMessage] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSending, setIsSending] = useState(false);

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
    if (!id) {
      navigate('/dashboard/messages');
      return;
    }

    // Mark messages as read when opening conversation
    const readMessages = async () => {
      if (id) {
        await markAsRead(id);
      }
    };

    const loadData = async () => {
      // Simulating network delay
      await new Promise(resolve => setTimeout(resolve, 500));
      setIsLoading(false);
      await readMessages();
    };

    loadData();
  }, [id, markAsRead, navigate]);

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [getConversationMessages, id]);

  const conversation = id ? getConversation(id) : undefined;
  const messages = id ? getConversationMessages(id) : [];

  const getClientInfo = () => {
    if (!user || !conversation) return { name: 'Unknown', avatar: undefined };

    const clientId = conversation.participants.find(pid => pid !== user.id);
    if (!clientId) return { name: 'Unknown', avatar: undefined };

    return clientData[clientId] || { name: 'Unknown', avatar: undefined };
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!id || !newMessage.trim()) return;

    setIsSending(true);
    try {
      await sendMessage(id, newMessage.trim());
      setNewMessage('');
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsSending(false);
    }
  };

  const formatMessageTime = (timestamp: string) => {
    const date = new Date(timestamp);

    // If it's today, show the time, otherwise show the date and time
    const now = new Date();
    if (
      date.getDate() === now.getDate() &&
      date.getMonth() === now.getMonth() &&
      date.getFullYear() === now.getFullYear()
    ) {
      return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
    } else {
      return `${date.toLocaleDateString('fr-FR')} ${date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}`;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-workit-purple"></div>
      </div>
    );
  }

  if (!conversation) {
    return (
      <div className="bg-red-500/10 text-red-500 p-4 rounded-md mb-6">
        <p>Conversation non trouvée</p>
        <Link to="/dashboard/messages" className="text-workit-purple hover:underline mt-2 inline-block">
          Retour aux messages
        </Link>
      </div>
    );
  }

  const client = getClientInfo();

  return (
    <div className="flex flex-col h-[calc(100vh-16rem)]">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <Link
            to="/dashboard/messages"
            className="mr-4 p-2 rounded-full bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white transition"
          >
            <ChevronLeft size={24} />
          </Link>
          <div className="flex items-center">
            <div className="w-12 h-12 rounded-full overflow-hidden mr-3">
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
            <div>
              <h2 className="text-xl font-semibold text-white">{client.name}</h2>
              <div className="flex items-center text-gray-400 text-sm">
                {conversation.title && (
                  <>
                    <Calendar size={14} className="mr-1" />
                    <span>{conversation.title}</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
        <div>
          <Link
            to="/dashboard/messages"
            className="px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-600 transition"
          >
            <ArrowLeft size={18} className="mr-2 inline-block" />
            Retour
          </Link>
        </div>
      </div>

      <div className="flex-1 bg-gray-900 rounded-lg border border-gray-800 mb-4 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center text-gray-400">
              <p>Aucun message dans cette conversation. Envoyez un message pour commencer.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => {
                const isOwn = user && message.senderId === user.id;

                return (
                  <div
                    key={message.id}
                    className={`flex ${isOwn ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className="max-w-[75%]">
                      <div
                        className={`rounded-lg px-4 py-3 ${
                          isOwn
                            ? 'bg-workit-purple text-white'
                            : 'bg-gray-800 text-gray-200'
                        }`}
                      >
                        <p className="whitespace-pre-wrap break-words">{message.content}</p>
                      </div>
                      <div
                        className={`flex items-center mt-1 text-xs ${
                          isOwn ? 'justify-end text-gray-400' : 'justify-start text-gray-500'
                        }`}
                      >
                        <span>{formatMessageTime(message.timestamp)}</span>
                        {isOwn && (
                          <span className="ml-2 flex items-center">
                            {message.isRead ? (
                              <Check size={14} className="text-workit-purple" />
                            ) : (
                              <Clock size={14} />
                            )}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="p-4 border-t border-gray-800">
          <form onSubmit={handleSendMessage} className="flex items-center">
            <input
              type="text"
              placeholder="Votre message..."
              className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-l-md text-white focus:outline-none focus:border-workit-purple"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              disabled={isSending}
            />
            <button
              type="submit"
              className="px-4 py-3 bg-workit-purple text-white rounded-r-md hover:bg-workit-purple-light transition disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isSending || !newMessage.trim()}
            >
              <Send size={20} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default DashboardMessageDetails;
