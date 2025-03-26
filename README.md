# WorkiT - Plateforme de Services Freelance

WorkiT est une plateforme de mise en relation entre freelances et clients, permettant aux talents de proposer leurs services dans divers domaines comme le développement web, le design, la rédaction, la traduction, et plus encore.

## Caractéristiques

- 🚀 Interface utilisateur moderne et responsive
- 👤 Système d'authentification complet (inscription, connexion, profil)
- 🔍 Recherche et filtrage des services
- 💼 Publication et gestion de services
- 📝 Système d'avis et évaluations
- 💰 Paiement sécurisé (simulé)
- 💬 Messagerie entre freelances et clients
- 👨‍💼 Section emploi pour les offres d'emploi

## Technologies utilisées

- **Frontend**: React, TypeScript, Tailwind CSS
- **Backend**: Node.js, Express (simulé côté client)
- **Base de données**: MongoDB (simulée côté client)
- **Authentification**: JWT (simulé côté client)
- **Routing**: React Router v7
- **Formulaires**: React Hook Form, Zod
- **Animations**: Framer Motion

## Installation et démarrage

1. Clonez ce dépôt
2. Installez les dépendances avec `bun install`
3. Lancez le serveur de développement avec `bun run dev`
4. Accédez à l'application via `http://localhost:5173`

## Structure du projet

```
src/
├── assets/         # Images, icônes et autres ressources
├── components/     # Composants React réutilisables
├── context/        # Context API pour la gestion d'état global
├── hooks/          # Hooks personnalisés
├── pages/          # Pages principales de l'application
├── types/          # Définitions de types TypeScript
└── utils/          # Fonctions utilitaires
```

## Déploiement

Le projet est configuré pour être déployé facilement sur Netlify.
La configuration se trouve dans le fichier `netlify.toml`.

## Licence

MIT
