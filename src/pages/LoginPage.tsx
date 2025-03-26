import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

const loginSchema = z.object({
  email: z.string().email({ message: 'Adresse email invalide' }),
  password: z.string().min(1, { message: 'Veuillez saisir votre mot de passe' }),
  rememberMe: z.boolean().optional(),
});

type LoginFormData = z.infer<typeof loginSchema>;

const LoginPage = () => {
  const { login } = useUser();
  const navigate = useNavigate();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  const { register, handleSubmit, formState: { errors } } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      rememberMe: false,
    }
  });

  const onSubmit = async (data: LoginFormData) => {
    try {
      setIsSubmitting(true);
      setLoginError(null);
      await login(data.email, data.password);
      navigate('/');
    } catch (error) {
      console.error('Login error:', error);
      setLoginError('Identifiants incorrects. Veuillez réessayer.');
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
          <h1 className="text-3xl font-bold text-white mb-2">Se Connecter</h1>
          <p className="text-gray-400">Bienvenue sur WorkiT</p>
        </div>

        <div className="bg-workit-dark-card rounded-lg p-8 shadow-xl">
          {loginError && (
            <div className="mb-4 p-3 bg-red-900 bg-opacity-30 border border-red-800 rounded text-red-400 text-sm">
              {loginError}
            </div>
          )}

          <form onSubmit={handleSubmit(onSubmit)}>
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

            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <label htmlFor="password" className="block text-white text-sm font-medium">
                  Mot de passe
                </label>
                <Link to="/forgot-password" className="text-workit-purple hover:underline text-sm">
                  Mot de passe oublié?
                </Link>
              </div>
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

            <div className="mb-6">
              <div className="flex items-center">
                <input
                  id="rememberMe"
                  type="checkbox"
                  {...register('rememberMe')}
                  className="h-4 w-4 rounded border-gray-600 bg-gray-700 text-workit-purple focus:ring-workit-purple"
                />
                <label htmlFor="rememberMe" className="ml-2 block text-sm text-gray-400">
                  Se souvenir de moi
                </label>
              </div>
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full bg-workit-purple text-white py-3 px-4 rounded-md font-medium hover:bg-workit-purple-light transition disabled:opacity-70"
            >
              {isSubmitting ? 'Connexion en cours...' : 'Se Connecter'}
            </button>
          </form>

          <div className="mt-6 text-center text-sm text-gray-400">
            Pas encore de compte? <Link to="/register" className="text-workit-purple hover:underline">S'inscrire</Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
