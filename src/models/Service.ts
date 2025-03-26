import mongoose, { Document, Schema } from 'mongoose';

// Define the interface for a service document
export interface IService extends Document {
  title: string;
  category: string;
  subcategory: string;
  price: number;
  description: string;
  features: string[];
  deliveryTime: number;
  revisions: string;
  userId: string;
  image?: string;
  gallery?: string[];
  createdAt: Date;
  updatedAt: Date;
}

// Export IService as Service for backward compatibility
export type Service = IService;

// Create the schema for services
const ServiceSchema = new Schema<IService>(
  {
    title: {
      type: String,
      required: [true, 'Le titre est requis'],
      trim: true,
      minlength: [5, 'Le titre doit contenir au moins 5 caractères'],
    },
    category: {
      type: String,
      required: [true, 'La catégorie est requise'],
      trim: true,
    },
    subcategory: {
      type: String,
      required: [true, 'La sous-catégorie est requise'],
      trim: true,
    },
    price: {
      type: Number,
      required: [true, 'Le prix est requis'],
      min: [0, 'Le prix ne peut pas être négatif'],
    },
    description: {
      type: String,
      required: [true, 'La description est requise'],
      trim: true,
      minlength: [20, 'La description doit contenir au moins 20 caractères'],
    },
    features: {
      type: [String],
      required: [true, 'Les fonctionnalités sont requises'],
      validate: {
        validator: (features: string[]) => features.length > 0,
        message: 'Au moins une fonctionnalité est requise',
      },
    },
    deliveryTime: {
      type: Number,
      required: [true, 'Le délai de livraison est requis'],
      min: [1, 'Le délai de livraison doit être d\'au moins 1 jour'],
    },
    revisions: {
      type: String,
      required: [true, 'Le nombre de révisions est requis'],
      trim: true,
    },
    userId: {
      type: String,
      required: [true, 'L\'ID de l\'utilisateur est requis'],
    },
    image: {
      type: String,
    },
    gallery: {
      type: [String],
    },
  },
  { timestamps: true }
);

// Create a text index for search functionality
ServiceSchema.index({
  title: 'text',
  description: 'text',
  category: 'text',
  subcategory: 'text'
});

// Export the model
const ServiceModel = mongoose.models.Service || mongoose.model<IService>('Service', ServiceSchema);

export default ServiceModel;
