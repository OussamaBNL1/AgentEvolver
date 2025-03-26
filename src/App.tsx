import { createBrowserRouter, RouterProvider, Outlet } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Navbar from './components/layout/Navbar';
import Footer from './components/layout/Footer';
import HomePage from './pages/HomePage';
import ServicesPage from './pages/ServicesPage';
import ServiceDetailsPage from './pages/ServiceDetailsPage';
import ProfilePage from './pages/ProfilePage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ContactPage from './pages/ContactPage';
import EmploymentPage from './pages/EmploymentPage';
import NotFoundPage from './pages/NotFoundPage';
import DashboardLayout from './components/layout/DashboardLayout';
import DashboardHome from './pages/dashboard/DashboardHome';
import DashboardServices from './pages/dashboard/DashboardServices';
import DashboardOrders from './pages/dashboard/DashboardOrders';
import DashboardAddService from './pages/dashboard/DashboardAddService';
import DashboardEditService from './pages/dashboard/DashboardEditService';
import DashboardOrderDetails from './pages/dashboard/DashboardOrderDetails';
import { UserProvider } from './context/UserContext';

const AppLayout = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
};

function App() {
  const router = createBrowserRouter([
    {
      path: "/",
      element: <AppLayout />,
      children: [
        { index: true, element: <HomePage /> },
        { path: "services", element: <ServicesPage /> },
        { path: "service/:id", element: <ServiceDetailsPage /> },
        { path: "profile", element: <ProfilePage /> },
        { path: "employment", element: <EmploymentPage /> },
        { path: "contact", element: <ContactPage /> },
        { path: "*", element: <NotFoundPage /> },
      ],
    },
    { path: "/login", element: <LoginPage /> },
    { path: "/register", element: <RegisterPage /> },
    {
      path: "/dashboard",
      element: <DashboardLayout><Outlet /></DashboardLayout>,
      children: [
        { index: true, element: <DashboardHome /> },
        { path: "services", element: <DashboardServices /> },
        { path: "services/add", element: <DashboardAddService /> },
        { path: "services/edit/:id", element: <DashboardEditService /> },
        { path: "orders", element: <DashboardOrders /> },
        { path: "orders/:id", element: <DashboardOrderDetails /> },
      ],
    },
  ]);

  return (
    <UserProvider>
      <RouterProvider router={router} />
    </UserProvider>
  );
}

export default App;
