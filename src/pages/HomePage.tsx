import { Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { useUser } from '../context/UserContext';

interface Service {
  id: string;
  title: string;
  category: string;
  subcategory: string;
  price: number;
  image: string;
  provider: {
    id: string;
    name: string;
    avatar?: string;
  };
}

const HomePage = () => {
  const { isAuthenticated } = useUser();
  const [trendingServices, setTrendingServices] = useState<Service[]>([]);

  useEffect(() => {
    // Mock trending services data
    const mockServices: Service[] = [
      {
        id: '1',
        title: 'Développement Web Fullstack - React, Node, MongoDB',
        category: 'Développement de sites web',
        subcategory: 'Développement Fullstack',
        price: 150,
        image: 'https://images.unsplash.com/photo-1587620962725-abab7fe55159?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1031&q=80',
        provider: {
          id: 'user1',
          name: 'David Martin',
          avatar: 'https://randomuser.me/api/portraits/men/32.jpg',
        },
      },
      {
        id: '2',
        title: 'Logo et Identité Visuelle pour votre Entreprise',
        category: 'Graphisme & Design',
        subcategory: 'Logo & Branding',
        price: 99,
        image: 'https://images.unsplash.com/photo-1611532736592-9f7d450fe20f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1374&q=80',
        provider: {
          id: 'user2',
          name: 'Sophie Dubois',
          avatar: 'https://randomuser.me/api/portraits/women/44.jpg',
        },
      },
      {
        id: '3',
        title: 'Traduction Professionnelle en Français, Anglais et Espagnol',
        category: 'Traduction',
        subcategory: 'Traduction Professionnelle',
        price: 50,
        image: 'https://images.unsplash.com/photo-1456406644174-8ddd4cd52a06?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1368&q=80',
        provider: {
          id: 'user3',
          name: 'Pierre Lambert',
          avatar: 'https://randomuser.me/api/portraits/men/64.jpg',
        },
      },
      {
        id: '4',
        title: 'Montage Vidéo Professionnel pour vos Projets',
        category: 'Vidéo & Animation',
        subcategory: 'Montage Vidéo',
        price: 120,
        image: 'https://images.unsplash.com/photo-1574717024453-354056de4834?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80',
        provider: {
          id: 'user4',
          name: 'Marie Leclerc',
          avatar: 'https://randomuser.me/api/portraits/women/33.jpg',
        },
      },
      {
        id: '5',
        title: 'Création d\'Applications Mobiles iOS et Android',
        category: 'Développement d\'applications mobiles',
        subcategory: 'Applications Mobiles',
        price: 200,
        image: 'https://images.unsplash.com/photo-1551650975-87deedd944c3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1374&q=80',
        provider: {
          id: 'user5',
          name: 'Thomas Bernard',
          avatar: 'https://randomuser.me/api/portraits/men/76.jpg',
        },
      },
      {
        id: '6',
        title: 'Rédaction Web SEO et Copywriting',
        category: 'Rédaction & Traduction',
        subcategory: 'Rédaction SEO',
        price: 80,
        image: 'https://images.unsplash.com/photo-1560785496-3c9d27877182?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1374&q=80',
        provider: {
          id: 'user6',
          name: 'Julie Moreau',
          avatar: 'https://randomuser.me/api/portraits/women/17.jpg',
        },
      },
    ];

    setTrendingServices(mockServices);
  }, []);

  return (
    <div>
      {/* Hero Section */}
      <section className="bg-workit-purple text-white py-16">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            Gagnez avec votre talent !
          </h1>
          <p className="text-lg mb-8 max-w-2xl mx-auto">
            Vos compétences sont précieuses. Commencez votre parcours
            avec nous et regardez chaque service se transformer en un
            flux constant de revenus.
          </p>
          <div className="mt-8">
            <button className="bg-black text-white font-medium py-3 px-6 rounded-md hover:bg-gray-800 transition">
              Monétiser vos compétences
            </button>
          </div>
        </div>
      </section>

      {/* Trending Services Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-2xl font-bold text-white">Services en Tendance</h2>
            <Link to="/services" className="text-workit-purple hover:underline">
              Tous les Services →
            </Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {trendingServices.map((service) => (
              <div key={service.id} className="bg-workit-dark-card rounded-lg overflow-hidden shadow-lg">
                <div className="relative">
                  <div className="absolute top-2 left-2 bg-workit-purple text-white text-xs font-semibold px-2 py-1 rounded">
                    Tendance
                  </div>
                  <div className="h-48 overflow-hidden">
                    <img
                      src={service.image}
                      alt={service.title}
                      className="w-full h-full object-cover transition duration-300 hover:scale-105"
                    />
                  </div>
                  <button className="absolute top-2 right-2 bg-black bg-opacity-60 rounded-full p-2 hover:bg-opacity-80">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5 text-white"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z" />
                    </svg>
                  </button>
                </div>
                <div className="p-4">
                  <div className="text-xs text-gray-400 mb-1">{service.category}</div>
                  <h3 className="font-semibold text-white mb-2 line-clamp-2">
                    <Link to={`/service/${service.id}`} className="hover:text-workit-purple">
                      {service.title}
                    </Link>
                  </h3>
                  <div className="flex items-center mb-4">
                    <div className="w-8 h-8 rounded-full overflow-hidden mr-2">
                      <img
                        src={service.provider.avatar}
                        alt={service.provider.name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <span className="text-gray-400 text-sm">{service.provider.name}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <div className="text-xs text-gray-400">à partir de:</div>
                    <div className="text-lg font-bold text-workit-purple">{service.price.toFixed(2)} TND</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-workit-purple py-16 text-white text-center">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold mb-4">Rejoignez notre communauté!</h2>
          <p className="mb-8 max-w-2xl mx-auto">
            Que vous soyez un freelance à la recherche de clients ou une
            entreprise cherchant les meilleurs talents, WorkiT est la plateforme
            qu'il vous faut.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link
              to="/register"
              className="inline-block bg-black text-white font-medium py-3 px-6 rounded-md hover:bg-gray-800 transition"
            >
              S'inscrire Maintenant
            </Link>
            <Link
              to="/services"
              className="inline-block bg-transparent border border-white text-white font-medium py-3 px-6 rounded-md hover:bg-white hover:text-workit-purple transition"
            >
              Explorer les Services
            </Link>
          </div>
        </div>
      </section>

      {/* Two Column Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="text-3xl font-bold text-white mb-6">
                Gagnez avec votre talent !
              </h2>
              <p className="text-gray-300 mb-6">
                Vos compétences sont précieuses. Commencez votre parcours
                avec nous et regardez chaque service se transformer en un
                flux constant de revenus.
              </p>
              <Link
                to={isAuthenticated ? "/profile" : "/register"}
                className="inline-block bg-workit-purple text-white font-medium py-3 px-6 rounded-md hover:bg-workit-purple-light transition"
              >
                Monétiser vos compétences
              </Link>
            </div>
            <div className="rounded-lg overflow-hidden">
              <img
                src="https://images.unsplash.com/photo-1531482615713-2afd69097998?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80"
                alt="Équipe de freelances qui travaillent"
                className="w-full h-auto"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Second Two Column Section */}
      <section className="py-16 bg-workit-purple">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <div className="order-2 md:order-1 rounded-lg overflow-hidden">
              <img
                src="https://images.unsplash.com/photo-1600880292203-757bb62b4baf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80"
                alt="Professionnels en réunion"
                className="w-full h-auto"
              />
            </div>
            <div className="order-1 md:order-2">
              <h2 className="text-3xl font-bold text-white mb-6">
                Recrutez les plus talentueux
              </h2>
              <p className="text-white mb-6">
                Travaillez avec le plus grand réseau de professionnels
                indépendants et accomplissez vos besoins dans des délais
                rapides.
              </p>
              <Link
                to="/services"
                className="inline-block bg-black text-white font-medium py-3 px-6 rounded-md hover:bg-gray-800 transition"
              >
                Trouver des Services
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
