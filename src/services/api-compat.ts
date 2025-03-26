// Compatibility layer for old API functions
// This file provides backward compatibility with the old API functions
// that are used in the components

import { serviceApi } from './serviceApi';

// Re-export of old function names that map to the new API functions
export const getAllServices = async () => {
  const result = await serviceApi.getServices();
  return result.services;
};

export const getServiceById = async (id: string) => {
  return await serviceApi.getServiceById(id);
};

export const getServicesByUserId = async (userId: string) => {
  return await serviceApi.getUserServices(userId);
};

export const createService = async (serviceData: any) => {
  return await serviceApi.createService(serviceData);
};

export const updateService = async (id: string, serviceData: any) => {
  // Assuming the user ID comes from the serviceData
  const userId = serviceData.userId || '';
  return await serviceApi.updateService(id, serviceData, userId);
};

export const deleteService = async (id: string, userId: string) => {
  return await serviceApi.deleteService(id, userId);
};

export const searchServices = async (query: string, category?: string) => {
  const result = await serviceApi.getServices(1, 20, category, query);
  return result.services;
};

export const getServicesByCategory = async (category: string) => {
  const result = await serviceApi.getServices(1, 20, category);
  return result.services;
};
