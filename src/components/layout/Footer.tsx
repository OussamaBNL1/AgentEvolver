import { Link } from 'react-router-dom';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-workit-dark py-10 mt-auto">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand Column */}
          <div>
            <Link to="/" className="flex items-center gap-2 text-white text-xl font-bold mb-4">
              <div className="flex">
                <div className="w-6 h-6 rounded-full bg-workit-purple-light"></div>
                <div className="w-6 h-6 rounded-full bg-workit-purple-light -ml-3"></div>
              </div>
              <span>WorkiT</span>
            </Link>
            <p className="text-gray-400 text-sm mb-6">
              WorkiT est la plateforme leader mettant en relation
              des freelances talentueux avec des clients tant au
              niveau local que mondial. Que vous soyez un
              acheteur recherchant des services professionnels
              ou un vendeur proposant vos compétences, notre
              marketplace répond à tous vos besoins.
            </p>
          </div>

          {/* Links Column */}
          <div>
            <h3 className="text-white font-semibold mb-4">WorkiT</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/about" className="text-gray-400 hover:text-white transition">
                  À propos de nous
                </Link>
              </li>
              <li>
                <Link to="/buyer-protection" className="text-gray-400 hover:text-white transition">
                  Protection des Acheteurs/Vendeurs
                </Link>
              </li>
              <li>
                <Link to="/approval-process" className="text-gray-400 hover:text-white transition">
                  Processus d'Approbation
                </Link>
              </li>
            </ul>
          </div>

          {/* Contact Column */}
          <div>
            <h3 className="text-white font-semibold mb-4">Contactez Nous</h3>
            <ul className="space-y-2">
              <li className="text-gray-400">
                <a
                  href="mailto:hi@workit.com"
                  className="hover:text-white transition"
                >
                  hi@workit.com
                </a>
              </li>
              <li>
                <a
                  href="https://facebook.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-white transition"
                >
                  Facebook
                </a>
              </li>
              <li>
                <a
                  href="https://linkedin.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-white transition"
                >
                  LinkedIn
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-8 pt-6 flex flex-col md:flex-row justify-between items-center">
          <div className="flex space-x-6 mb-4 md:mb-0">
            <Link to="/terms" className="text-gray-400 text-sm hover:text-white transition">
              Termes et Conditions
            </Link>
            <Link to="/privacy" className="text-gray-400 text-sm hover:text-white transition">
              Politique de Confidentialité
            </Link>
          </div>
          <p className="text-gray-400 text-sm">
            © {currentYear} WorkiT. Tous droits réservés.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
