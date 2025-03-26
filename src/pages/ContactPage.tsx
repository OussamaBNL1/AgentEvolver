import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { useUser } from '../context/UserContext';

const contactSchema = z.object({
  fullName: z.string().min(3, { message: 'Le nom doit contenir au moins 3 caractères' }),
  email: z.string().email({ message: 'Adresse email invalide' }),
  subject: z.string().min(1, { message: 'Veuillez sélectionner un sujet' }),
  message: z.string().min(10, { message: 'Le message doit contenir au moins 10 caractères' }),
  acceptPrivacy: z.boolean().refine(val => val === true, { message: 'Vous devez accepter la politique de confidentialité' }),
  userType: z.enum(['client', 'freelancer']),
});

type ContactFormData = z.infer<typeof contactSchema>;

const ContactPage = () => {
  const { user } = useUser();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const { register, handleSubmit, reset, setValue, watch, formState: { errors } } = useForm<ContactFormData>({
    resolver: zodResolver(contactSchema),
    defaultValues: {
      fullName: user?.name || '',
      email: user?.email || '',
      subject: '',
      message: '',
      acceptPrivacy: false,
      userType: 'client',
    }
  });

  const userType = watch('userType');

  const onSubmit = async (data: ContactFormData) => {
    try {
      setIsSubmitting(true);
      // In a real app, this would be an API call to send the contact form
      console.log('Contact form submitted:', data);

      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      setIsSuccess(true);
      reset();
    } catch (error) {
      console.error('Contact form submission error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="py-12 px-4">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">Nous sommes là pour vous aider</h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Remplissez le formulaire ci-dessous et nous vous répondrons dans les plus brefs délais.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Left Column - Contact Info */}
          <div className="bg-workit-dark-card rounded-lg p-6">
            <h2 className="text-xl font-semibold text-white mb-6">
              Informations de Contact
            </h2>
            <div className="space-y-6">
              <div>
                <h3 className="text-workit-purple font-medium mb-2 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" />
                    <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" />
                  </svg>
                  Email
                </h3>
                <p className="text-gray-400">
                  <a href="mailto:hi@workit.com" className="hover:text-workit-purple">
                    hi@workit.com
                  </a>
                </p>
              </div>

              <div>
                <h3 className="text-workit-purple font-medium mb-2 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                  </svg>
                  Adresse
                </h3>
                <p className="text-gray-400">
                  123 Rue de l'Innovation<br />
                  75001 Paris, France
                </p>
              </div>

              <div>
                <h3 className="text-workit-purple font-medium mb-2 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M2 3a1 1 0 011-1h2.153a1 1 0 01.986.836l.74 4.435a1 1 0 01-.54 1.06l-1.548.773a11.037 11.037 0 006.105 6.105l.774-1.548a1 1 0 011.059-.54l4.435.74a1 1 0 01.836.986V17a1 1 0 01-1 1h-2C7.82 18 2 12.18 2 5V3z" />
                  </svg>
                  Heures d'Ouverture
                </h3>
                <p className="text-gray-400">
                  Lundi - Vendredi: 9h00 - 18h00<br />
                  Samedi - Dimanche: Fermé
                </p>
              </div>
            </div>

            <div className="mt-8">
              <h3 className="text-white font-medium mb-4">Suivez-nous</h3>
              <div className="flex space-x-4">
                <a
                  href="https://facebook.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-workit-purple transition"
                >
                  Facebook
                </a>
                <a
                  href="https://linkedin.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-workit-purple transition"
                >
                  LinkedIn
                </a>
                <a
                  href="https://twitter.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-workit-purple transition"
                >
                  Twitter
                </a>
              </div>
            </div>

            <div className="mt-10">
              <h3 className="text-white font-medium mb-4">Nos Fonctionnalités</h3>
              <ul className="space-y-3">
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-workit-purple mr-2 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                  </svg>
                  <div>
                    <h4 className="text-white text-sm font-medium">Rôles Flexibles</h4>
                    <p className="text-gray-400 text-sm">
                      Choisissez d'être Freelancer ou Employeur,
                      avec possibilité de changer.
                    </p>
                  </div>
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-workit-purple mr-2 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clipRule="evenodd" />
                  </svg>
                  <div>
                    <h4 className="text-white text-sm font-medium">Chat en Temps Réel</h4>
                    <p className="text-gray-400 text-sm">
                      Messagerie directe avec partage de fichiers
                      pour une collaboration fluide.
                    </p>
                  </div>
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-workit-purple mr-2 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M4 4a2 2 0 00-2 2v1h16V6a2 2 0 00-2-2H4z" />
                    <path fillRule="evenodd" d="M18 9H2v5a2 2 0 002 2h12a2 2 0 002-2V9zM4 13a1 1 0 011-1h1a1 1 0 110 2H5a1 1 0 01-1-1zm5-1a1 1 0 100 2h1a1 1 0 100-2H9z" clipRule="evenodd" />
                  </svg>
                  <div>
                    <h4 className="text-white text-sm font-medium">Paiements Sécurisés</h4>
                    <p className="text-gray-400 text-sm">
                      Supporte la carte E-Dinar et le paiement à la
                      livraison pour des transactions sûres.
                    </p>
                  </div>
                </li>
              </ul>
            </div>
          </div>

          {/* Right Column - Contact Form */}
          <div className="md:col-span-2 bg-workit-dark-card rounded-lg p-6">
            <h2 className="text-xl font-semibold text-white mb-6">
              Envoyez-nous un Message
            </h2>

            {isSuccess ? (
              <div className="bg-green-900 bg-opacity-30 border border-green-800 rounded p-4 text-green-400">
                <p className="font-medium">Message envoyé avec succès!</p>
                <p className="mt-1">Nous vous répondrons dans les plus brefs délais.</p>
                <button
                  onClick={() => setIsSuccess(false)}
                  className="mt-3 bg-workit-purple text-white px-4 py-2 rounded-md text-sm hover:bg-workit-purple-light transition"
                >
                  Envoyer un autre message
                </button>
              </div>
            ) : (
              <form onSubmit={handleSubmit(onSubmit)}>
                <div className="flex space-x-3 mb-6">
                  <button
                    type="button"
                    onClick={() => setValue('userType', 'client')}
                    className={`flex-1 py-3 px-4 rounded-md text-center font-medium transition ${
                      userType === 'client'
                        ? 'bg-workit-purple text-white'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                  >
                    Je suis un Client
                  </button>
                  <button
                    type="button"
                    onClick={() => setValue('userType', 'freelancer')}
                    className={`flex-1 py-3 px-4 rounded-md text-center font-medium transition ${
                      userType === 'freelancer'
                        ? 'bg-workit-purple text-white'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                  >
                    Je suis un Freelancer
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  <div>
                    <label htmlFor="fullName" className="block text-white text-sm font-medium mb-2">
                      Nom Complet <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      id="fullName"
                      {...register('fullName')}
                      className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                      placeholder="Votre nom"
                    />
                    {errors.fullName && (
                      <p className="mt-1 text-red-500 text-xs">{errors.fullName.message}</p>
                    )}
                  </div>

                  <div>
                    <label htmlFor="email" className="block text-white text-sm font-medium mb-2">
                      Email <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="email"
                      id="email"
                      {...register('email')}
                      className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                      placeholder="votre@email.com"
                    />
                    {errors.email && (
                      <p className="mt-1 text-red-500 text-xs">{errors.email.message}</p>
                    )}
                  </div>
                </div>

                <div className="mb-4">
                  <label htmlFor="subject" className="block text-white text-sm font-medium mb-2">
                    Sujet <span className="text-red-500">*</span>
                  </label>
                  <select
                    id="subject"
                    {...register('subject')}
                    className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                  >
                    <option value="">Sélectionnez un sujet</option>
                    <option value="question-generale">Question Générale</option>
                    <option value="support-technique">Support Technique</option>
                    <option value="suggestion">Suggestion</option>
                    <option value="partenariat">Partenariat</option>
                    <option value="signaler-probleme">Signaler un Problème</option>
                  </select>
                  {errors.subject && (
                    <p className="mt-1 text-red-500 text-xs">{errors.subject.message}</p>
                  )}
                </div>

                <div className="mb-6">
                  <label htmlFor="message" className="block text-white text-sm font-medium mb-2">
                    Message <span className="text-red-500">*</span>
                  </label>
                  <textarea
                    id="message"
                    {...register('message')}
                    rows={5}
                    className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                    placeholder="Comment pouvons-nous vous aider?"
                  ></textarea>
                  {errors.message && (
                    <p className="mt-1 text-red-500 text-xs">{errors.message.message}</p>
                  )}
                </div>

                <div className="mb-6">
                  <div className="flex items-start">
                    <div className="flex items-center h-5">
                      <input
                        id="acceptPrivacy"
                        type="checkbox"
                        {...register('acceptPrivacy')}
                        className="h-4 w-4 rounded border-gray-600 bg-gray-700 text-workit-purple focus:ring-workit-purple"
                      />
                    </div>
                    <div className="ml-3 text-sm">
                      <label htmlFor="acceptPrivacy" className="text-gray-300">
                        J'accepte que mes données soient traitées conformément à la <a href="/privacy" className="text-workit-purple hover:underline">politique de confidentialité</a>.
                      </label>
                      {errors.acceptPrivacy && (
                        <p className="mt-1 text-red-500 text-xs">{errors.acceptPrivacy.message}</p>
                      )}
                    </div>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full bg-workit-purple text-white py-3 px-4 rounded-md font-medium hover:bg-workit-purple-light transition disabled:opacity-70"
                >
                  {isSubmitting ? 'Envoi en cours...' : 'Envoyer le Message'}
                </button>
              </form>
            )}
          </div>
        </div>

        <div className="mt-16">
          <h2 className="text-2xl font-bold text-white mb-6 text-center">Questions Fréquentes</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-workit-dark-card p-6 rounded-lg">
              <h3 className="text-workit-purple font-semibold mb-2">Comment changer de rôle sur WorkiT?</h3>
              <p className="text-gray-400">
                Vous pouvez facilement basculer entre les rôles de Freelancer et d'Employeur dans votre tableau de
                bord, sous "Paramètres du compte".
              </p>
            </div>

            <div className="bg-workit-dark-card p-6 rounded-lg">
              <h3 className="text-workit-purple font-semibold mb-2">Quelles méthodes de paiement acceptez-vous?</h3>
              <p className="text-gray-400">
                Nous acceptons actuellement les paiements par carte E-Dinar et les paiements à la livraison pour
                assurer des transactions sécurisées.
              </p>
            </div>

            <div className="bg-workit-dark-card p-6 rounded-lg">
              <h3 className="text-workit-purple font-semibold mb-2">Comment puis-je contacter un freelancer?</h3>
              <p className="text-gray-400">
                Une fois que vous avez trouvé un freelancer qui vous intéresse, vous pouvez utiliser notre système
                de messagerie intégré pour lui envoyer un message direct.
              </p>
            </div>

            <div className="bg-workit-dark-card p-6 rounded-lg">
              <h3 className="text-workit-purple font-semibold mb-2">Comment sont gérés les litiges entre utilisateurs?</h3>
              <p className="text-gray-400">
                Notre équipe de support traite tous les litiges. Vous pouvez signaler un problème via la page de contact
                ou directement depuis votre interface de commande.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContactPage;
