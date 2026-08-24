# Frontend Dockerfile
FROM node:22-alpine

WORKDIR /app

ARG NEXT_PUBLIC_API_URL=http://localhost:8001/api
ARG NEXT_PUBLIC_APP_URL=http://localhost:3002
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_PUBLIC_APP_URL=${NEXT_PUBLIC_APP_URL}

# Copy package files
COPY package.json package-lock.json* ./

# Install the exact audited dependency tree.
RUN npm ci

# Copy source
COPY . .

# Build Next.js app
RUN npm run build

# Expose port
EXPOSE 3002

# Start app
CMD ["npm", "start"]
