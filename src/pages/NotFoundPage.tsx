import { Link } from 'react-router-dom';

const NotFoundPage = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] px-4 text-center">
      <h1 className="text-workit-purple text-9xl font-bold mb-4">404</h1>
      <h2 className="text-white text-2xl md:text-3xl font-semibold mb-6">
        Page Non Trouvée
      </h2>
      <p className="text-gray-400 max-w-md mb-8">
        La page que vous recherchez n'existe pas ou a été déplacée.
        Veuillez vérifier l'URL ou retourner à la page d'accueil.
      </p>
      <Link
        to="/"
        className="bg-workit-purple px-6 py-3 rounded-md text-white font-medium hover:bg-workit-purple-light transition"
      >
        Retourner à l'Accueil
      </Link>
    </div>
  );
};

export default NotFoundPage;
