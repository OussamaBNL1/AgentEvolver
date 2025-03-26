import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface User {
  id: string;
  name: string;
  email: string;
  bio?: string;
  location?: string;
  skills?: string[];
  profilePicture?: string;
  isFreelancer?: boolean;
  createdAt: string;
}

interface UserContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (userData: Partial<User> & { password: string }) => Promise<void>;
  logout: () => void;
  updateUser: (userData: Partial<User>) => Promise<void>;
  updateProfilePicture: (imageFile: File) => Promise<void>;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export const useUser = () => {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

export const UserProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);

  useEffect(() => {
    // Check if user is logged in from localStorage
    const storedUser = localStorage.getItem('workit_user');
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setUser(parsedUser);
        setIsAuthenticated(true);
      } catch (error) {
        console.error('Error parsing stored user:', error);
        localStorage.removeItem('workit_user');
      }
    }
  }, []);

  const login = async (email: string, password: string) => {
    // In a real app, this would be an API call to authenticate
    // Simulating successful login for now
    const mockUser: User = {
      id: 'user_' + Math.random().toString(36).substring(2, 9),
      name: email.split('@')[0],
      email,
      createdAt: new Date().toISOString(),
    };

    setUser(mockUser);
    setIsAuthenticated(true);
    localStorage.setItem('workit_user', JSON.stringify(mockUser));
  };

  const register = async (userData: Partial<User> & { password: string }) => {
    // In a real app, this would be an API call to register
    // Simulating successful registration for now
    const { password, ...userDataWithoutPassword } = userData;

    const mockUser: User = {
      id: 'user_' + Math.random().toString(36).substring(2, 9),
      name: userData.name || '',
      email: userData.email || '',
      createdAt: new Date().toISOString(),
      ...userDataWithoutPassword,
    };

    setUser(mockUser);
    setIsAuthenticated(true);
    localStorage.setItem('workit_user', JSON.stringify(mockUser));
  };

  const logout = () => {
    setUser(null);
    setIsAuthenticated(false);
    localStorage.removeItem('workit_user');
  };

  const updateUser = async (userData: Partial<User>) => {
    if (!user) return;

    const updatedUser = { ...user, ...userData };
    setUser(updatedUser);
    localStorage.setItem('workit_user', JSON.stringify(updatedUser));
  };

  const updateProfilePicture = async (imageFile: File) => {
    if (!user) return;

    // In a real app, this would upload the image to a server
    // Simulating a successful upload by creating a data URL
    const reader = new FileReader();
    reader.onload = () => {
      const updatedUser = {
        ...user,
        profilePicture: reader.result as string
      };

      setUser(updatedUser);
      localStorage.setItem('workit_user', JSON.stringify(updatedUser));
    };

    reader.readAsDataURL(imageFile);
  };

  return (
    <UserContext.Provider
      value={{
        user,
        isAuthenticated,
        login,
        register,
        logout,
        updateUser,
        updateProfilePicture
      }}
    >
      {children}
    </UserContext.Provider>
  );
};
