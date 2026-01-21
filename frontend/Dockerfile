# Frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source
COPY . .

# Build Next.js app
RUN npm run build

# Expose port
EXPOSE 3002

# Start app
CMD ["npm", "start"]
