# kinemats
Personal library to analyze kinematic time series in Python (dim_reduction, classification, etc.). Mostly relying on quaternions.


## Use as  Git submodules

Allow to bring other git repositories (*submodules*) within a main git project (*superproject*)

`git submodule add <GIT URL> [folder name]`

When cloning a repository that has submodules, these submodule folders are empty and need to be checkout:

`git submodule update --init`

To update all the submodules to their respective master branch:

`git submodule update --remote`

Summary of the submodules from the superproject

`git submodule status`

**To keep in mind:**
- Once your `cd` inside a submodule, all git commands are with respect to the submodule and not the superproject.
- If there is a change in the submodule from the superproject, first you need to `commit` the submodule repo and **then** commit the superproject with a message like "Update reference to submodule". The superproject just keeps a reference to the commit in the submodule that needs to be `checkout`, does not track all the changes like it would do to the other folders.
