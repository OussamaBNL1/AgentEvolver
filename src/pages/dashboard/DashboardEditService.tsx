import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useUser } from '../../context/UserContext';
import { getServiceById, updateService } from '../../services/api-compat';
import { ServiceData } from '../../services/serviceApi';

interface ServiceWithId extends ServiceData {
  _id: string;
}

const DashboardEditService = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useUser();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [service, setService] = useState<ServiceWithId | null>(null);
  const [formData, setFormData] = useState<Partial<ServiceData>>({
    title: '',
    description: '',
    price: 0,
    category: '',
    subcategory: '',
    features: [],
    deliveryTime: 1,
    revisions: '1',
  });

  useEffect(() => {
    const fetchService = async () => {
      if (!id) return;

      setIsLoading(true);
      setError(null);

      try {
        const serviceData = await getServiceById(id);

        if (serviceData) {
          setService(serviceData);
          setFormData({
            title: serviceData.title,
            description: serviceData.description,
            price: serviceData.price,
            category: serviceData.category,
            subcategory: serviceData.subcategory || '',
            features: serviceData.features || [],
            deliveryTime: serviceData.deliveryTime || 1,
            revisions: serviceData.revisions || '1',
          });
        } else {
          setError('Service not found');
        }
      } catch (err) {
        setError('Error loading service data');
        console.error('Error loading service data:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchService();
  }, [id]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'price' || name === 'deliveryTime' ? parseFloat(value) : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user || !service || !id) return;

    try {
      setIsLoading(true);
      await updateService(id, {...formData, userId: user.id});
      navigate('/dashboard/services');
    } catch (err) {
      setError('Error updating service');
      console.error('Error updating service:', err);
    } finally {
      setIsLoading(false);
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

  if (!service) {
    return (
      <div className="bg-yellow-500/10 text-yellow-500 p-4 rounded-md mb-6">
        <p>Service non trouvé</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-bold text-white">Modifier le service</h1>
        <button
          onClick={() => navigate('/dashboard/services')}
          className="px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-600 transition"
        >
          Retour
        </button>
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="title" className="block text-white mb-2">
              Titre
            </label>
            <input
              type="text"
              id="title"
              name="title"
              value={formData.title}
              onChange={handleChange}
              className="w-full p-3 rounded-md bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-workit-purple"
              required
            />
          </div>

          <div className="mb-4">
            <label htmlFor="description" className="block text-white mb-2">
              Description
            </label>
            <textarea
              id="description"
              name="description"
              value={formData.description}
              onChange={handleChange}
              rows={6}
              className="w-full p-3 rounded-md bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-workit-purple"
              required
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label htmlFor="price" className="block text-white mb-2">
                Prix (TND)
              </label>
              <input
                type="number"
                id="price"
                name="price"
                value={formData.price}
                onChange={handleChange}
                min="0"
                step="0.01"
                className="w-full p-3 rounded-md bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-workit-purple"
                required
              />
            </div>

            <div>
              <label htmlFor="category" className="block text-white mb-2">
                Catégorie
              </label>
              <select
                id="category"
                name="category"
                value={formData.category}
                onChange={handleChange}
                className="w-full p-3 rounded-md bg-gray-800 border border-gray-700 text-white focus:outline-none focus:border-workit-purple"
                required
              >
                <option value="">Sélectionner une catégorie</option>
                <option value="web">Développement Web</option>
                <option value="mobile">Développement Mobile</option>
                <option value="design">Design</option>
                <option value="marketing">Marketing</option>
                <option value="writing">Rédaction</option>
                <option value="other">Autre</option>
              </select>
            </div>
          </div>

          <div className="mt-6 flex space-x-4">
            <button
              type="submit"
              className="px-6 py-3 bg-workit-purple text-white rounded-md hover:bg-workit-purple-light transition"
            >
              Mettre à jour
            </button>
            <button
              type="button"
              onClick={() => navigate('/dashboard/services')}
              className="px-6 py-3 bg-gray-700 text-white rounded-md hover:bg-gray-600 transition"
            >
              Annuler
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default DashboardEditService;
