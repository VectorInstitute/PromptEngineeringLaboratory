# Setting up your Repository Clone

If you are new to git and GitHub there are a lot of introductions, including on the [github documentation website](https://docs.github.com/en/get-started/start-your-journey/hello-world).

1. In your terminal, navigate to wherever you'd like to store your working clone of the repository

2. `git clone git@github.com:VectorInstitute/PromptEngineeringLaboratory.git`

    Creates a local copy of the repository on your own machine.

3. `git branch`

    You have a main and can create branches of your own to work off of and merge to main. __NOTE__: You should always create a branch to make your personal changes, especially if you would like to share such changes with someone on your team, as you will __not__ be able to push your code directly to main.

4. `git checkout -b <branch_name>`

    Create a branch off of whatever branch you are currently on, including main.

5. Make some changes and commits

    This can be done using the `git add <file path>` and `git commit -m "Some Message"` command workflow, or using an integrated source control tool like that in VS code.

6. `git push origin <branch_name>`

    Makes your code changes available on github. This allows others to merge changes to your branch and take a look at your code on their own machine. It is also the first step in merging code to other branches that you would like to add to.

    __NOTE__: If it is your first time pushing your branch, you need to include an additional option `git push -u origin <branch name>` to move it from your machine to github for the first time.

7. If there are changes on another branch that you want to incorporate into your own branches, you need to follow these steps. An example is when there is an update to the main branch and you want those changes to be reflected in your code.

    1) `git checkout <merge source branch>`: Move onto the branch you would like to merge into your branch.

    2) `git pull` will update the local version of the branch you are currently on with the changes on remote.

    3) `git checkout <target branch for merge>`: Move onto the branch you would like to update.

    4) `git merge <merge source branch>`: Merge the `<merge source branch>` code into your branch code.

    5) Follow the merge instructions to commit the merge to your branch.

# Proposing changes to the Main Repository

Changes to the main repository branch are done through pull requests. If you're new to pull requests, some documentation is [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

For more information on contributing directly to the repository, see `CONTRIBUTING.md`.
