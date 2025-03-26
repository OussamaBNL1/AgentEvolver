import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

const jobSchema = z.object({
  title: z.string().min(5, { message: 'Le titre doit contenir au moins 5 caractères' }),
  company: z.string().min(2, { message: 'Le nom de l\'entreprise est requis' }),
  location: z.string().min(2, { message: 'La localisation est requise' }),
  type: z.string().min(1, { message: 'Le type de contrat est requis' }),
  salary: z.string().min(1, { message: 'Le salaire est requis' }),
  description: z.string().min(20, { message: 'La description doit contenir au moins 20 caractères' }),
  requirements: z.string().min(10, { message: 'Les prérequis sont requis' }),
});

type JobFormData = z.infer<typeof jobSchema>;

const PostJobPage = () => {
  const { isAuthenticated, user } = useUser();
  const navigate = useNavigate();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);

  const { register, handleSubmit, formState: { errors } } = useForm<JobFormData>({
    resolver: zodResolver(jobSchema),
    defaultValues: {
      company: user?.name || '',
      location: user?.location || '',
      type: 'Temps plein',
    }
  });

  // Redirect if not authenticated
  if (!isAuthenticated) {
    navigate('/login', { state: { from: '/post-job' } });
    return null;
  }

  const onSubmit = async (data: JobFormData) => {
    setIsSubmitting(true);

    try {
      // In a real app, this would be an API call to save the job listing
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API delay

      // Parse requirements into an array
      const requirementsArray = data.requirements
        .split('\n')
        .filter(line => line.trim().length > 0)
        .map(line => line.trim());

      // Create job object
      const newJob = {
        id: `job_${Math.random().toString(36).substr(2, 9)}`,
        title: data.title,
        company: data.company,
        location: data.location,
        type: data.type,
        salary: data.salary,
        description: data.description,
        requirements: requirementsArray,
        postedAt: new Date().toISOString().split('T')[0],
        postedBy: user?.id,
      };

      // In a real app, we would store this in a database
      console.log('New job created:', newJob);

      // Show success message
      setSubmitSuccess(true);

      // Redirect after 2 seconds
      setTimeout(() => {
        navigate('/employment');
      }, 2000);

    } catch (error) {
      console.error('Error posting job:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (submitSuccess) {
    return (
      <div className="py-12 px-4">
        <div className="container mx-auto max-w-3xl">
          <div className="bg-workit-dark-card rounded-lg p-8 text-center">
            <div className="w-16 h-16 bg-green-500 bg-opacity-20 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-white mb-4">Offre d'emploi publiée avec succès!</h1>
            <p className="text-gray-400 mb-6">
              Votre offre d'emploi a été publiée avec succès. Vous allez être redirigé vers la page des emplois.
            </p>
            <div className="animate-pulse text-workit-purple text-sm">Redirection en cours...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="py-12 px-4">
      <div className="container mx-auto max-w-3xl">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-white mb-2">Publier une offre d'emploi</h1>
          <p className="text-gray-400">
            Remplissez le formulaire ci-dessous pour publier votre offre d'emploi
          </p>
        </div>

        <div className="bg-workit-dark-card rounded-lg overflow-hidden">
          <form onSubmit={handleSubmit(onSubmit)} className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div className="md:col-span-2">
                <label htmlFor="title" className="block text-white text-sm font-medium mb-2">
                  Titre du poste*
                </label>
                <input
                  id="title"
                  type="text"
                  {...register('title')}
                  placeholder="ex: Développeur Frontend React"
                  className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                />
                {errors.title && (
                  <p className="mt-1 text-red-500 text-xs">{errors.title.message}</p>
                )}
              </div>

              <div>
                <label htmlFor="company" className="block text-white text-sm font-medium mb-2">
                  Nom de l'entreprise*
                </label>
                <input
                  id="company"
                  type="text"
                  {...register('company')}
                  className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                />
                {errors.company && (
                  <p className="mt-1 text-red-500 text-xs">{errors.company.message}</p>
                )}
              </div>

              <div>
                <label htmlFor="location" className="block text-white text-sm font-medium mb-2">
                  Localisation*
                </label>
                <input
                  id="location"
                  type="text"
                  {...register('location')}
                  placeholder="ex: Paris, France"
                  className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                />
                {errors.location && (
                  <p className="mt-1 text-red-500 text-xs">{errors.location.message}</p>
                )}
              </div>

              <div>
                <label htmlFor="type" className="block text-white text-sm font-medium mb-2">
                  Type de contrat*
                </label>
                <select
                  id="type"
                  {...register('type')}
                  className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                >
                  <option value="Temps plein">Temps plein</option>
                  <option value="Temps partiel">Temps partiel</option>
                  <option value="Freelance">Freelance</option>
                  <option value="Stage">Stage</option>
                  <option value="Alternance">Alternance</option>
                  <option value="CDD">CDD</option>
                  <option value="CDI">CDI</option>
                </select>
                {errors.type && (
                  <p className="mt-1 text-red-500 text-xs">{errors.type.message}</p>
                )}
              </div>

              <div>
                <label htmlFor="salary" className="block text-white text-sm font-medium mb-2">
                  Salaire*
                </label>
                <input
                  id="salary"
                  type="text"
                  {...register('salary')}
                  placeholder="ex: 45 000€ - 55 000€ / an"
                  className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                />
                {errors.salary && (
                  <p className="mt-1 text-red-500 text-xs">{errors.salary.message}</p>
                )}
              </div>

              <div className="md:col-span-2">
                <label htmlFor="description" className="block text-white text-sm font-medium mb-2">
                  Description du poste*
                </label>
                <textarea
                  id="description"
                  {...register('description')}
                  rows={6}
                  placeholder="Décrivez le poste, les responsabilités, l'entreprise..."
                  className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                />
                {errors.description && (
                  <p className="mt-1 text-red-500 text-xs">{errors.description.message}</p>
                )}
              </div>

              <div className="md:col-span-2">
                <label htmlFor="requirements" className="block text-white text-sm font-medium mb-2">
                  Prérequis et compétences* (une par ligne)
                </label>
                <textarea
                  id="requirements"
                  {...register('requirements')}
                  rows={5}
                  placeholder="Expérience en développement React&#10;Connaissance de TypeScript&#10;Expérience avec Git&#10;etc."
                  className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                />
                {errors.requirements && (
                  <p className="mt-1 text-red-500 text-xs">{errors.requirements.message}</p>
                )}
              </div>
            </div>

            <div className="border-t border-gray-800 pt-6 flex justify-end">
              <button
                type="submit"
                disabled={isSubmitting}
                className="bg-workit-purple text-white px-6 py-3 rounded-md font-medium hover:bg-workit-purple-light transition disabled:opacity-70 inline-flex items-center"
              >
                {isSubmitting ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Publication en cours...
                  </>
                ) : (
                  'Publier l\'offre d\'emploi'
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default PostJobPage;
