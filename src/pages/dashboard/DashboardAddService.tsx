import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useUser } from '../../context/UserContext';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { ChevronLeft, Upload, Plus, X, AlertCircle } from 'lucide-react';

// Define schema for service creation
const serviceSchema = z.object({
  title: z.string().min(10, { message: 'Le titre doit contenir au moins 10 caractères' }),
  category: z.string().min(1, { message: 'La catégorie est requise' }),
  subcategory: z.string().min(1, { message: 'La sous-catégorie est requise' }),
  price: z.coerce.number().positive({ message: 'Le prix doit être positif' }),
  description: z.string().min(50, { message: 'La description doit contenir au moins 50 caractères' }),
  deliveryTime: z.coerce.number().int().min(1, { message: 'Le délai de livraison est requis' }),
  revisions: z.string().min(1, { message: 'Le nombre de révisions est requis' }),
});

type ServiceFormData = z.infer<typeof serviceSchema>;

const DashboardAddService = () => {
  const { user } = useUser();
  const navigate = useNavigate();
  const [isSaving, setIsSaving] = useState(false);
  const [mainImage, setMainImage] = useState<File | null>(null);
  const [mainImagePreview, setMainImagePreview] = useState<string | null>(null);
  const [galleryImages, setGalleryImages] = useState<File[]>([]);
  const [galleryPreviews, setGalleryPreviews] = useState<string[]>([]);
  const [features, setFeatures] = useState<string[]>(['']);
  const [featureInput, setFeatureInput] = useState('');
  const [categories, setCategories] = useState<{ id: string; name: string; subcategories: {id: string; name: string}[] }[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('');

  // Setup form
  const { register, handleSubmit, formState: { errors }, setValue, watch } = useForm<ServiceFormData>({
    resolver: zodResolver(serviceSchema),
    defaultValues: {
      title: '',
      category: '',
      subcategory: '',
      price: 0,
      description: '',
      deliveryTime: 7,
      revisions: 'unlimited',
    }
  });

  const watchCategory = watch('category');

  // Load categories
  useEffect(() => {
    // Simulate API call to get categories
    const mockCategories = [
      {
        id: 'dev',
        name: 'Développement',
        subcategories: [
          { id: 'web', name: 'Développement Web' },
          { id: 'mobile', name: 'Développement Mobile' },
          { id: 'desktop', name: 'Applications Desktop' },
          { id: 'api', name: 'API & Backend' },
          { id: 'ecommerce', name: 'E-commerce' },
          { id: 'wordpress', name: 'WordPress' },
        ]
      },
      {
        id: 'design',
        name: 'Design',
        subcategories: [
          { id: 'logo', name: 'Logo & Branding' },
          { id: 'web-design', name: 'Web Design' },
          { id: 'ui-ux', name: 'UI/UX Design' },
          { id: 'illustrations', name: 'Illustrations' },
          { id: 'social-media', name: 'Médias Sociaux' },
        ]
      },
      {
        id: 'marketing',
        name: 'Marketing',
        subcategories: [
          { id: 'seo', name: 'SEO' },
          { id: 'social', name: 'Médias Sociaux' },
          { id: 'ads', name: 'Publicité' },
          { id: 'email', name: 'Email Marketing' },
          { id: 'content', name: 'Marketing de Contenu' },
        ]
      },
      {
        id: 'writing',
        name: 'Rédaction & Traduction',
        subcategories: [
          { id: 'copywriting', name: 'Copywriting' },
          { id: 'translation', name: 'Traduction' },
          { id: 'content', name: 'Rédaction de Contenu' },
          { id: 'proofreading', name: 'Correction & Relecture' },
        ]
      },
      {
        id: 'video',
        name: 'Vidéo & Animation',
        subcategories: [
          { id: 'editing', name: 'Montage Vidéo' },
          { id: 'animation', name: 'Animation' },
          { id: 'motion', name: 'Motion Graphics' },
          { id: 'intro', name: 'Intros & Outros' },
        ]
      },
    ];

    setCategories(mockCategories);
  }, []);

  // Handle category change
  useEffect(() => {
    if (watchCategory) {
      setSelectedCategory(watchCategory);
      setValue('subcategory', '');
    }
  }, [watchCategory, setValue]);

  // Handle main image upload
  const handleMainImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setMainImage(file);

    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setMainImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  // Handle gallery images upload
  const handleGalleryImagesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    // Limit to 5 images total
    const newFiles = Array.from(files).slice(0, 5 - galleryImages.length);
    setGalleryImages(prev => [...prev, ...newFiles]);

    // Create previews
    newFiles.forEach(file => {
      const reader = new FileReader();
      reader.onload = () => {
        setGalleryPreviews(prev => [...prev, reader.result as string]);
      };
      reader.readAsDataURL(file);
    });
  };

  // Remove gallery image
  const removeGalleryImage = (index: number) => {
    setGalleryImages(prev => prev.filter((_, i) => i !== index));
    setGalleryPreviews(prev => prev.filter((_, i) => i !== index));
  };

  // Handle features add/remove
  const addFeature = () => {
    if (featureInput.trim() && features.length < 10) {
      setFeatures(prev => [...prev, featureInput.trim()]);
      setFeatureInput('');
    }
  };

  const removeFeature = (index: number) => {
    setFeatures(prev => prev.filter((_, i) => i !== index));
  };

  // Handle form submission
  const onSubmit = async (data: ServiceFormData) => {
    try {
      setIsSaving(true);

      // Validate images
      if (!mainImage) {
        alert("L'image principale est requise");
        setIsSaving(false);
        return;
      }

      // Filter out empty features
      const validFeatures = features.filter(f => f.trim() !== '');

      // This would be an API call in a real app
      console.log('Form data:', {
        ...data,
        features: validFeatures,
        mainImage,
        galleryImages,
      });

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Redirect to services page on success
      navigate('/dashboard/services');
    } catch (error) {
      console.error('Error creating service:', error);
      setIsSaving(false);
    }
  };

  return (
    <div>
      <div className="mb-8">
        <Link
          to="/dashboard/services"
          className="inline-flex items-center text-gray-400 hover:text-white transition"
        >
          <ChevronLeft size={16} className="mr-1" />
          <span>Retour aux services</span>
        </Link>
      </div>

      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Ajouter un service</h1>
          <p className="text-gray-400">
            Créez un nouveau service à proposer à vos clients.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
        {/* Title and Price */}
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <h2 className="text-xl font-semibold text-white mb-4">Informations de base</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2">
              <label className="block text-white text-sm font-medium mb-2">
                Titre du service <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                {...register('title')}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                placeholder="Ex: Développement de site web responsive avec React"
              />
              {errors.title && (
                <p className="mt-1 text-red-500 text-xs">{errors.title.message}</p>
              )}
            </div>

            <div>
              <label className="block text-white text-sm font-medium mb-2">
                Prix (TND) <span className="text-red-500">*</span>
              </label>
              <input
                type="number"
                min="0"
                step="0.01"
                {...register('price')}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                placeholder="Ex: 150"
              />
              {errors.price && (
                <p className="mt-1 text-red-500 text-xs">{errors.price.message}</p>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div>
              <label className="block text-white text-sm font-medium mb-2">
                Catégorie <span className="text-red-500">*</span>
              </label>
              <select
                {...register('category')}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
              >
                <option value="">Sélectionner une catégorie</option>
                {categories.map(category => (
                  <option key={category.id} value={category.id}>
                    {category.name}
                  </option>
                ))}
              </select>
              {errors.category && (
                <p className="mt-1 text-red-500 text-xs">{errors.category.message}</p>
              )}
            </div>

            <div>
              <label className="block text-white text-sm font-medium mb-2">
                Sous-catégorie <span className="text-red-500">*</span>
              </label>
              <select
                {...register('subcategory')}
                disabled={!selectedCategory}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-workit-purple disabled:opacity-60"
              >
                <option value="">Sélectionner une sous-catégorie</option>
                {selectedCategory &&
                  categories
                    .find(c => c.id === selectedCategory)
                    ?.subcategories.map(subcat => (
                      <option key={subcat.id} value={subcat.id}>
                        {subcat.name}
                      </option>
                    ))
                }
              </select>
              {errors.subcategory && (
                <p className="mt-1 text-red-500 text-xs">{errors.subcategory.message}</p>
              )}
            </div>
          </div>
        </div>

        {/* Description */}
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <h2 className="text-xl font-semibold text-white mb-4">Description</h2>

          <div>
            <label className="block text-white text-sm font-medium mb-2">
              Description détaillée <span className="text-red-500">*</span>
            </label>
            <textarea
              {...register('description')}
              rows={6}
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
              placeholder="Décrivez en détail ce que vous proposez, ce que le client obtiendra, et comment vous travaillez..."
            ></textarea>
            {errors.description && (
              <p className="mt-1 text-red-500 text-xs">{errors.description.message}</p>
            )}
          </div>
        </div>

        {/* Features */}
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <h2 className="text-xl font-semibold text-white mb-4">Ce qui est inclus</h2>

          <div className="space-y-4 mb-6">
            {features.filter(f => f.trim() !== '').map((feature, index) => (
              <div key={index} className="flex items-center">
                <div className="flex-1 bg-gray-800 rounded-md px-4 py-2 text-white">
                  {feature}
                </div>
                <button
                  type="button"
                  onClick={() => removeFeature(index)}
                  className="ml-2 text-red-500 hover:text-red-400"
                >
                  <X size={20} />
                </button>
              </div>
            ))}
          </div>

          <div className="flex">
            <input
              type="text"
              value={featureInput}
              onChange={(e) => setFeatureInput(e.target.value)}
              className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-l-md text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
              placeholder="Ajouter une fonctionnalité incluse..."
            />
            <button
              type="button"
              onClick={addFeature}
              disabled={features.length >= 10 || !featureInput.trim()}
              className="bg-workit-purple text-white px-4 py-2 rounded-r-md hover:bg-workit-purple-light transition disabled:opacity-50"
            >
              <Plus size={20} />
            </button>
          </div>

          <p className="mt-2 text-gray-400 text-sm">
            {features.filter(f => f.trim() !== '').length}/10 fonctionnalités ajoutées
          </p>
        </div>

        {/* Delivery Details */}
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <h2 className="text-xl font-semibold text-white mb-4">Détails de livraison</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-white text-sm font-medium mb-2">
                Délai de livraison (jours) <span className="text-red-500">*</span>
              </label>
              <input
                type="number"
                min="1"
                max="90"
                {...register('deliveryTime')}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
              />
              {errors.deliveryTime && (
                <p className="mt-1 text-red-500 text-xs">{errors.deliveryTime.message}</p>
              )}
            </div>

            <div>
              <label className="block text-white text-sm font-medium mb-2">
                Nombre de révisions <span className="text-red-500">*</span>
              </label>
              <select
                {...register('revisions')}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
              >
                <option value="unlimited">Illimitées</option>
                <option value="1">1 révision</option>
                <option value="2">2 révisions</option>
                <option value="3">3 révisions</option>
                <option value="5">5 révisions</option>
                <option value="10">10 révisions</option>
              </select>
              {errors.revisions && (
                <p className="mt-1 text-red-500 text-xs">{errors.revisions.message}</p>
              )}
            </div>
          </div>
        </div>

        {/* Images */}
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <h2 className="text-xl font-semibold text-white mb-4">Images</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-white text-sm font-medium mb-2">
                Image principale <span className="text-red-500">*</span>
              </label>
              <div className="mt-2">
                {mainImagePreview ? (
                  <div className="relative rounded-lg overflow-hidden h-48">
                    <img
                      src={mainImagePreview}
                      alt="Preview"
                      className="w-full h-full object-cover"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setMainImage(null);
                        setMainImagePreview(null);
                      }}
                      className="absolute top-2 right-2 bg-black bg-opacity-70 text-white p-1 rounded-full hover:bg-opacity-90"
                    >
                      <X size={20} />
                    </button>
                  </div>
                ) : (
                  <div className="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center">
                    <input
                      type="file"
                      id="mainImage"
                      accept="image/*"
                      onChange={handleMainImageChange}
                      className="hidden"
                    />
                    <label
                      htmlFor="mainImage"
                      className="flex flex-col items-center justify-center cursor-pointer"
                    >
                      <Upload size={32} className="text-gray-500 mb-2" />
                      <span className="text-gray-400">Cliquez pour télécharger</span>
                      <span className="text-gray-500 text-sm">PNG, JPG, GIF (Max. 5MB)</span>
                    </label>
                  </div>
                )}
              </div>
            </div>

            <div>
              <label className="block text-white text-sm font-medium mb-2">
                Galerie d'images (optionnel)
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-2">
                {galleryPreviews.map((preview, index) => (
                  <div key={index} className="relative rounded-lg overflow-hidden h-24">
                    <img
                      src={preview}
                      alt={`Gallery ${index}`}
                      className="w-full h-full object-cover"
                    />
                    <button
                      type="button"
                      onClick={() => removeGalleryImage(index)}
                      className="absolute top-1 right-1 bg-black bg-opacity-70 text-white p-1 rounded-full hover:bg-opacity-90"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}

                {galleryPreviews.length < 5 && (
                  <div className="border-2 border-dashed border-gray-700 rounded-lg flex items-center justify-center h-24">
                    <input
                      type="file"
                      id="galleryImages"
                      accept="image/*"
                      multiple
                      onChange={handleGalleryImagesChange}
                      className="hidden"
                    />
                    <label
                      htmlFor="galleryImages"
                      className="flex flex-col items-center justify-center cursor-pointer w-full h-full"
                    >
                      <Plus size={24} className="text-gray-500" />
                    </label>
                  </div>
                )}
              </div>
              <p className="mt-2 text-gray-400 text-sm">
                {galleryPreviews.length}/5 images ajoutées
              </p>
            </div>
          </div>
        </div>

        {/* Legal Notice */}
        <div className="bg-gray-800 bg-opacity-50 border border-gray-700 rounded-lg p-4 flex items-start">
          <AlertCircle size={20} className="text-yellow-500 mr-3 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-gray-300">
            <p>En publiant ce service, vous confirmez :</p>
            <ul className="list-disc pl-5 mt-2 space-y-1 text-gray-400">
              <li>Que le contenu est votre propriété ou que vous avez les droits nécessaires pour l'utiliser</li>
              <li>Que vous respectez les conditions d'utilisation de WorkiT</li>
              <li>Que vous êtes responsable de la qualité du service fourni</li>
            </ul>
          </div>
        </div>

        {/* Submit Buttons */}
        <div className="flex justify-end space-x-4">
          <Link
            to="/dashboard/services"
            className="px-6 py-3 bg-gray-800 text-white rounded-md hover:bg-gray-700 transition"
          >
            Annuler
          </Link>
          <button
            type="submit"
            disabled={isSaving}
            className="px-6 py-3 bg-workit-purple text-white rounded-md hover:bg-workit-purple-light transition disabled:opacity-70"
          >
            {isSaving ? 'Création en cours...' : 'Créer le service'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default DashboardAddService;
