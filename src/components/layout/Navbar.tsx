import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useUser } from '../../context/UserContext';

const Navbar = () => {
  const { user, isAuthenticated, logout } = useUser();
  const navigate = useNavigate();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <nav className="bg-workit-purple p-4">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center">
          <Link to="/" className="flex items-center gap-2 text-white text-xl font-bold">
            <div className="flex">
              <div className="w-8 h-8 rounded-full bg-workit-purple-light"></div>
              <div className="w-8 h-8 rounded-full bg-workit-purple-light -ml-4"></div>
            </div>
            <span>WorkiT</span>
          </Link>
        </div>

        {/* Navigation links */}
        <div className="hidden md:flex space-x-6">
          <Link to="/" className="text-white hover:text-gray-200">
            Accueil
          </Link>
          <Link to="/services" className="text-white hover:text-gray-200">
            Services
          </Link>
          <Link to="/employment" className="text-white hover:text-gray-200">
            Emploi
          </Link>
          <Link to="/contact" className="text-white hover:text-gray-200">
            Contact
          </Link>
        </div>

        {/* Authentication buttons */}
        <div className="hidden md:flex items-center space-x-4">
          {isAuthenticated ? (
            <div className="flex items-center space-x-4 relative">
              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="flex items-center space-x-2"
              >
                <div className="w-8 h-8 rounded-full bg-workit-purple-dark text-white flex items-center justify-center">
                  {user?.profilePicture ? (
                    <img
                      src={user.profilePicture}
                      alt={user.name}
                      className="w-full h-full object-cover rounded-full"
                    />
                  ) : (
                    <span>{user?.name.charAt(0).toLowerCase()}</span>
                  )}
                </div>
                {user?.name && (
                  <span className="text-white">{user.name}</span>
                )}
              </button>

              {isMenuOpen && (
                <div className="absolute right-0 top-10 mt-2 w-48 bg-workit-dark-card rounded-md shadow-lg z-50">
                  <div className="py-1">
                    <Link
                      to="/profile"
                      className="block px-4 py-2 text-white hover:bg-gray-800"
                      onClick={() => setIsMenuOpen(false)}
                    >
                      Mon Profil
                    </Link>
                    <button
                      onClick={() => {
                        handleLogout();
                        setIsMenuOpen(false);
                      }}
                      className="block w-full text-left px-4 py-2 text-white hover:bg-gray-800"
                    >
                      Se Déconnecter
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <>
              <Link
                to="/login"
                className="text-white border border-white rounded-md px-4 py-1 hover:bg-workit-purple-dark transition"
              >
                Se Connecter
              </Link>
              <Link
                to="/register"
                className="bg-workit-purple-dark text-white rounded-md px-4 py-1 hover:bg-workit-purple-light transition"
              >
                Inscrivez
              </Link>
            </>
          )}
        </div>

        {/* Mobile menu button */}
        <button
          className="md:hidden flex items-center"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6 text-white"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
        </button>
      </div>

      {/* Mobile menu */}
      {isMenuOpen && (
        <div className="md:hidden bg-workit-purple-dark mt-2 p-4 rounded-lg">
          <div className="flex flex-col space-y-3">
            <Link
              to="/"
              className="text-white hover:text-gray-200"
              onClick={() => setIsMenuOpen(false)}
            >
              Accueil
            </Link>
            <Link
              to="/services"
              className="text-white hover:text-gray-200"
              onClick={() => setIsMenuOpen(false)}
            >
              Services
            </Link>
            <Link
              to="/employment"
              className="text-white hover:text-gray-200"
              onClick={() => setIsMenuOpen(false)}
            >
              Emploi
            </Link>
            <Link
              to="/contact"
              className="text-white hover:text-gray-200"
              onClick={() => setIsMenuOpen(false)}
            >
              Contact
            </Link>

            {isAuthenticated ? (
              <>
                <Link
                  to="/profile"
                  className="text-white hover:text-gray-200"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Mon Profil
                </Link>
                <button
                  onClick={() => {
                    handleLogout();
                    setIsMenuOpen(false);
                  }}
                  className="text-left text-white hover:text-gray-200"
                >
                  Se Déconnecter
                </button>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  className="text-white hover:text-gray-200"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Se Connecter
                </Link>
                <Link
                  to="/register"
                  className="text-white hover:text-gray-200"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Inscrivez
                </Link>
              </>
            )}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
