# Contributor Guide

## Welcome to the BitMind Subnet Contributor Community!

We're excited to have you interested in contributing to the BitMind Subnet. This guide aims to provide all the information you need to start contributing effectively to our project. Whether you're fixing bugs, adding features, or improving documentation, your help is welcome!

### How to Contribute

#### 1. Get Started
Before you start contributing, make sure to go through our [Setup Guide 🔧](docs/Setup.md) to get your development environment ready. Also, familiarize yourself with our [Project Structure and Terminology 📖](docs/Glossary.md) to understand the layout and terminology used throughout the project.

#### 2. Find an Issue
Browse through our [GitHub Issues](https://github.com/bitmind-ai/bitmind-subnet/issues) to find tasks that need help.

#### 3. Fork and Clone the Repository
- Fork the repository by clicking the "Fork" button on the top right of the page. Then, clone your fork to your local machine:
- Clone your fork to your local machine:
```bash
git clone https://github.com/your-username/bitmind-subnet.git
cd bitmind-subnet
```
- Set the original repository as your 'upstream' remote:
```bash
git remote add upstream https://github.com/bitmind-ai/bitmind-subnet.git
```
#### 4. Sync Your Fork.
Before you start making changes, sync your fork with the upstream repository to ensure you have the latest updates:
```bash
git fetch upstream
git checkout main
git merge upstream/main
```

#### 5. Create a Branch
Create a new branch to work on. It's best to name the branch something descriptive:
```
git checkout -b feature/add-new-detection-model
```

#### 6. Make Your Changes
Make changes to the codebase or documentation. Ensure you follow our coding standards (PEP-8) and write tests if you are adding or modifying functionality.

#### 7. Commit Your Changes
Keep your commits as small as possible and focused on a single aspect of improvement. This approach makes it easier to review and manage:
```bash
git add .
git commit -m "Add a detailed commit message describing the change"
```

#### 8. Push Your Changes
Push your changes to your fork:
```bash
git push origin feature/add-new-detection-model
```

####  9. Submit a Pull Request (PR)
Go to the Pull Requests tab in the original repository and click "New pull request". Compare branches and make sure you are proposing changes from your branch to the main repository's main branch. Provide a concise description of the changes and reference any related issues.

#### 10. Participate in the Code Review Process
Once your PR is submitted, other contributors and maintainers will review your changes. Engage in discussions and make any requested adjustments. Your contributions will be merged once they are approved.

#### Code of Conduct
We expect all contributors to adhere to our [Code of Conduct 📜](Code_of_Conduct.md), ensuring respect and productive collaboration. Please read Code of Conduct to understand the expectations for behavior.

####  Need Help?
If you have any questions or need further guidance, don't hesitate to ask for help in our Discord community. We're here to make your contribution process as smooth as possible!

Thank you for contributing to the BitMind Subnet! We appreciate your effort to help us improve and extend our capabilities in detecting AI-generated media.