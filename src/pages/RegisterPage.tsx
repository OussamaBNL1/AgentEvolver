import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

const registerSchema = z.object({
  name: z.string().min(3, { message: 'Le nom doit contenir au moins 3 caractères' }),
  email: z.string().email({ message: 'Adresse email invalide' }),
  password: z.string().min(8, { message: 'Le mot de passe doit contenir au moins 8 caractères' }),
  confirmPassword: z.string(),
  isFreelancer: z.boolean().optional(),
  acceptTerms: z.boolean().refine(val => val === true, { message: 'Vous devez accepter les conditions d\'utilisation' }),
}).refine(data => data.password === data.confirmPassword, {
  message: 'Les mots de passe ne correspondent pas',
  path: ['confirmPassword'],
});

type RegisterFormData = z.infer<typeof registerSchema>;

const RegisterPage = () => {
  const { register: registerUser } = useUser();
  const navigate = useNavigate();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { register, handleSubmit, formState: { errors } } = useForm<RegisterFormData>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      isFreelancer: false,
      acceptTerms: false,
    }
  });

  const onSubmit = async (data: RegisterFormData) => {
    try {
      setIsSubmitting(true);
      await registerUser({
        name: data.name,
        email: data.email,
        password: data.password,
        isFreelancer: data.isFreelancer,
      });
      navigate('/');
    } catch (error) {
      console.error('Registration error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-workit-dark px-4 py-12">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2 mb-6">
            <div className="flex">
              <div className="w-8 h-8 rounded-full bg-workit-purple-light"></div>
              <div className="w-8 h-8 rounded-full bg-workit-purple-light -ml-4"></div>
            </div>
            <span className="text-2xl font-bold text-white">WorkiT</span>
          </Link>
          <h1 className="text-3xl font-bold text-white mb-2">Créer un compte</h1>
          <p className="text-gray-400">Rejoignez la communauté WorkiT dès aujourd'hui</p>
        </div>

        <div className="bg-workit-dark-card rounded-lg p-8 shadow-xl">
          <form onSubmit={handleSubmit(onSubmit)}>
            <div className="mb-4">
              <label htmlFor="name" className="block text-white text-sm font-medium mb-2">
                Nom complet
              </label>
              <input
                type="text"
                id="name"
                {...register('name')}
                className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                placeholder="Jean Dupont"
              />
              {errors.name && (
                <p className="mt-1 text-red-500 text-xs">{errors.name.message}</p>
              )}
            </div>

            <div className="mb-4">
              <label htmlFor="email" className="block text-white text-sm font-medium mb-2">
                Email
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

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label htmlFor="password" className="block text-white text-sm font-medium mb-2">
                  Mot de passe
                </label>
                <input
                  type="password"
                  id="password"
                  {...register('password')}
                  className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                  placeholder="********"
                />
                {errors.password && (
                  <p className="mt-1 text-red-500 text-xs">{errors.password.message}</p>
                )}
              </div>

              <div>
                <label htmlFor="confirmPassword" className="block text-white text-sm font-medium mb-2">
                  Confirmer le mot de passe
                </label>
                <input
                  type="password"
                  id="confirmPassword"
                  {...register('confirmPassword')}
                  className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-workit-purple"
                  placeholder="********"
                />
                {errors.confirmPassword && (
                  <p className="mt-1 text-red-500 text-xs">{errors.confirmPassword.message}</p>
                )}
              </div>
            </div>

            <div className="mb-4">
              <div className="flex items-start">
                <div className="flex items-center h-5">
                  <input
                    id="isFreelancer"
                    type="checkbox"
                    {...register('isFreelancer')}
                    className="h-4 w-4 rounded border-gray-600 bg-gray-700 text-workit-purple focus:ring-workit-purple"
                  />
                </div>
                <div className="ml-3 text-sm">
                  <label htmlFor="isFreelancer" className="text-gray-300">
                    Je veux proposer mes services en tant que freelance
                  </label>
                  <p className="text-gray-500 text-xs mt-1">
                    En cochant cette case, vous pourrez proposer vos services sur la plateforme.
                  </p>
                </div>
              </div>
            </div>

            <div className="mb-6">
              <div className="flex items-start">
                <div className="flex items-center h-5">
                  <input
                    id="acceptTerms"
                    type="checkbox"
                    {...register('acceptTerms')}
                    className="h-4 w-4 rounded border-gray-600 bg-gray-700 text-workit-purple focus:ring-workit-purple"
                  />
                </div>
                <div className="ml-3 text-sm">
                  <label htmlFor="acceptTerms" className="text-gray-300">
                    J'accepte les <Link to="/terms" className="text-workit-purple hover:underline">conditions d'utilisation</Link> et la <Link to="/privacy" className="text-workit-purple hover:underline">politique de confidentialité</Link>
                  </label>
                  {errors.acceptTerms && (
                    <p className="mt-1 text-red-500 text-xs">{errors.acceptTerms.message}</p>
                  )}
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full bg-workit-purple text-white py-3 px-4 rounded-md font-medium hover:bg-workit-purple-light transition disabled:opacity-70"
            >
              {isSubmitting ? 'Création en cours...' : 'Créer un compte'}
            </button>
          </form>

          <div className="mt-6 text-center text-sm text-gray-400">
            Vous avez déjà un compte? <Link to="/login" className="text-workit-purple hover:underline">Se connecter</Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RegisterPage;
