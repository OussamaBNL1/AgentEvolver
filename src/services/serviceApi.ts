import axios from 'axios';
import { IService } from '../models/Service';

// Define service data type for create/update operations
export interface ServiceData {
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
}

// Define job data type for create/update operations
export interface JobData {
  title: string;
  description: string;
  category: string;
  budget: number;
  deadline: Date;
  skills: string[];
  location?: string;
  isRemote?: boolean;
  status?: 'open' | 'in-progress' | 'completed' | 'cancelled';
  userId: string;
  attachments?: string[];
}

// Create an axios instance with base URL
const API_URL = 'http://localhost:5001/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Service API functions
export const serviceApi = {
  // Get all services with filtering, sorting, and pagination
  async getServices(
    page = 1,
    limit = 10,
    category?: string,
    search?: string,
    sort?: string
  ) {
    try {
      const params = {
        page,
        limit,
        ...(category && { category }),
        ...(search && { search }),
        ...(sort && { sort }),
      };

      const response = await api.get('/services', { params });
      return response.data;
    } catch (error) {
      console.error('Error fetching services:', error);
      throw error;
    }
  },

  // Get a service by ID
  async getServiceById(id: string) {
    try {
      const response = await api.get(`/services/${id}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching service:', error);
      throw error;
    }
  },

  // Create a new service
  async createService(serviceData: ServiceData) {
    try {
      const response = await api.post('/services', serviceData);
      return response.data;
    } catch (error) {
      console.error('Error creating service:', error);
      throw error;
    }
  },

  // Update a service
  async updateService(id: string, serviceData: Partial<ServiceData>, userId: string) {
    try {
      const response = await api.put(`/services/${id}`, {
        ...serviceData,
        userId,
      });
      return response.data;
    } catch (error) {
      console.error('Error updating service:', error);
      throw error;
    }
  },

  // Delete a service
  async deleteService(id: string, userId: string) {
    try {
      const response = await api.delete(`/services/${id}`, {
        data: { userId },
      });
      return response.data;
    } catch (error) {
      console.error('Error deleting service:', error);
      throw error;
    }
  },

  // Get services by user ID
  async getUserServices(userId: string) {
    try {
      const response = await api.get(`/services/user/${userId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching user services:', error);
      throw error;
    }
  },

  // Helper function to convert error messages to a user-friendly format
  formatError(error: unknown) {
    if (axios.isAxiosError(error) && error.response) {
      return error.response.data.message || 'An error occurred';
    }
    return error instanceof Error ? error.message : 'Network error';
  },
};

// Job API functions
export const jobApi = {
  // Get all jobs with filtering, sorting, and pagination
  async getJobs(
    page = 1,
    limit = 10,
    category?: string,
    search?: string,
    sort?: string
  ) {
    try {
      const params = {
        page,
        limit,
        ...(category && { category }),
        ...(search && { search }),
        ...(sort && { sort }),
      };

      const response = await api.get('/jobs', { params });
      return response.data;
    } catch (error) {
      console.error('Error fetching jobs:', error);
      throw error;
    }
  },

  // Get a job by ID
  async getJobById(id: string) {
    try {
      const response = await api.get(`/jobs/${id}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching job:', error);
      throw error;
    }
  },

  // Create a new job
  async createJob(jobData: JobData) {
    try {
      const response = await api.post('/jobs', jobData);
      return response.data;
    } catch (error) {
      console.error('Error creating job:', error);
      throw error;
    }
  },

  // Update a job
  async updateJob(id: string, jobData: Partial<JobData>, userId: string) {
    try {
      const response = await api.put(`/jobs/${id}`, {
        ...jobData,
        userId,
      });
      return response.data;
    } catch (error) {
      console.error('Error updating job:', error);
      throw error;
    }
  },

  // Delete a job
  async deleteJob(id: string, userId: string) {
    try {
      const response = await api.delete(`/jobs/${id}`, {
        data: { userId },
      });
      return response.data;
    } catch (error) {
      console.error('Error deleting job:', error);
      throw error;
    }
  },

  // Get jobs by user ID
  async getUserJobs(userId: string) {
    try {
      const response = await api.get(`/jobs/user/${userId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching user jobs:', error);
      throw error;
    }
  },
};
