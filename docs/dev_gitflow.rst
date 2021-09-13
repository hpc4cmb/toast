.. _dev_gitflow:

Using Git for Development
#################################

We use the git distributed version control system for development.  Git is incredibly powerful compared to older centralized version control systems like CVS or Subversion.  In particular, git allows developers to make progress independently while still being able to coalesce their work into a functioning product at the end.  In addition to the underlying git version control system, we make extensive use of github features such as pull requests and "actions", which we use for continuous integration testing.

There is extensive online documentation about how to use git itself, and also how to use tools provided by github.  Here we focus on just the best practices we try to use in TOAST development.

.. note:: Many git documents describe the use of "git pull" or "git pull --rebase", which under-the-hood actually do a "fetch" followed by either a "merge" or "rebase".  This document avoids those shortcuts to make it more clear what is happening.

Starting Out
*******************

Unless you are on the core development team, you cannot push directly to the primary TOAST git repo.  Instead, you should "fork" the TOAST repo into your own github account.  Then clone that repo locally and add the primary TOAST repo as the "upstream" remote.  See `the github documentation <https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks>`_ for more details about using forks.  For the rest of this section, we assume that you have your fork set up and you have a local git clone of this repo (i.e. the "origin" remote points to your fork) and that the "upstream" remote points to the main TOAST repo.

.. note:: Make sure to set up your ssh keys with github and run an ssh agent, keychain, or similar tool on your system so that you do not need to type a password for every command that communicates with github.

Staying Up to Date
======================

As you work on branches in your local clone, you should periodically synchronize the main development branch between the upstream repo and your forked repo.  You can do this with:

.. code-block:: bash
    # Switch to your local checkout of main
    git checkout main
    # Fetch a copy of all the branches in your fork.
    # Also delete any copies of branches that no
    # longer exist in your fork.
    git fetch origin
    git remote prune origin
    # Fetch a copy of all the branches upstream, and
    # prune any branches that were deleted upstream.
    git fetch upstream
    git remote prune upstream
    # Rebase your local checkout of main against
    # your fork.
    git rebase origin/main
    # Rebase your local checkout of main against
    # the upstream version
    git rebase upstream/main
    # Push your up-to-date copy of main to your fork
    git push origin main

This sequence should always work, since you should never be committing things directly to main.  All work should occur in a branch (see below).  Since the above set of commands is tedious, you can create a shell alias for these (except for the first line)if you like.  For example, putting this in ``~/.bashrc`` (name it whatever you like):

.. code-block:: bash
    git-sync-main () {
        # Check that we are on the main branch so that we
        # do not rebase on the wrong local branch
        current=$(git branch --show-current)
        if [ "x${current}" != "xmain" ]; then
            echo "You must checkout the main branch before running this command"
        else
            git fetch origin
            git remote prune origin
            git fetch upstream
            git remote prune upstream
            git rebase origin/main
            git rebase upstream/main
            git push origin main
        fi
    }

Now you can synchronize your local copy of main (and also keep your clone directory tidy) with:

.. code-block:: bash
    git checkout main
    git-sync-main


Working on a Branch
==========================

Whether you are hacking on a long-term development project or making a small pull request to fix a bug, you should work on a branch.

- setting up the branch

- making commits:  set EDITOR to what you want

You should periodically rebase your branch off of main in order to keep up to date with other upstream changes, and also to resolve any conflicts as they come up- rather than dealing with them all at once at the end when you open a pull request.  First, commit any changes on your branch and then synchronize your local checkout of main (using the shell function described previously, or manually):

.. code-block:: bash
    git checkout main
    git-sync-main

Now rebase your branch against your local checkout of main:

.. code-block:: bash
    git checkout mybranch
    git rebase main
    # If there are some small conflicts, open the conflicting files
    # in an editor and search for "<<<<<" and then edit it as needed.
    # Note that both the Atom editor and VS Code have graphical
    # displays that let you easily select which one you want (or both)
    # with a single click.  Then save the files and mark them resolved
    # and continue.
    git add path/to/file
    git add other/file/with/conflicts
    git rebase --continue
    # Eventually all of your commits will be replayed and the rebase
    # will be done.  If things get really crazy, do
    git rebase --abort
    # And see the section, "When Things Go Wrong" below.
    # Now we can force-push our copy of mybranch back to the origin
    git push -f origin mybranch


- why rebase instead of merge:  linear history, fewer useless merge commits, easier to follow what changes are being introduced.

- when to merge:  "permanent" or very long-lived branches, merging feature branches into main.



When Things Go Wrong
**************************

No matter how much experience we have with git, sometimes bad things hkappen.  This might be from accidental work flow errors in our own local checkout or it might be due to ustream changes that introduce big conflicts with our local work.  Here is a sequence of steps that can be attempted to fix the situation.

Repeated Conflicts During Rebase
=======================================

Sometimes attempting to rebase a branch against main causes repeated conflicts.  This can happen if you have merged main into you local branch and then made modifications to that code.  During the rebase, each commit is replayed onto the tip of main.  You may resolve one conflict only to have it reappear on the next round of ``git rebase --continue``.  This issue can often be resolved by aborting the rebase and first squashing all the changes in your branch into a single commit.  This way all that matters is the change from the starting point to the end point, rather than intermediate commits that make one change followed by later commits that undo that change.

If you are new to doing interactive rebasing (in order to squash), then I recommend first making a copy of your working branch:

.. code-block:: bash
    # Switch to your branch
    git checkout mybranch
    # Checkout a new branch that is a copy of this
    git checkout -b mybranch_copy

Now type ``git log`` and find the commit hash that was just before your first commit.  Copy this hash.  Now interactively rebase your branch against this starting point:

.. code-block:: bash
    git rebase -i <commit hash before your first commit>
    # Follow instructions and in your editor, mark the first
    # commit in the list as one to keep, and mark the rest
    # with "S" for squash.  Save and exit your editor.
    # Now edit the full commit message for your one big
    # commit.  You can rephrase your commit messages or just
    # make them into a big list.  Save and exit your editor.

Now if you type ``git log`` you will see all of your work as one big commit with your new detailed commit message.  Now that your work is contained in one large commit, we can try to rebase this against main:

.. code-block:: bash
    git rebase main

Now resolve any conflicts.  You will only have to resolve these once, since there is only one commit now.  If you started this work on a copy of your original branch, now you can push your copied branch and open a new pull request from this copy.  Once it is merged, you can clean up both the copy and the original from your local clone.


Last Resort
=================

Sometimes you may be dealing with a situation where the branch you are working with is very out of date, or includes a mixture of rebasing and merging against main.  This garbles the history and makes it challenging or impossible to cleanly rebase against main.  In some cases, such a branch may even include unintentional reverting of changes in main from other upstream merges.

- manually go through the full diff and make sure all changes are intended, and not actually reverting things that it should not.  Merge this into main, even if it makes the history a mess.

- copy files with changes into a temporary location, and then make a new branch from main.  Copy the changed files into the new branch and do any cleanups.  Delete the original branch.


Conclusion
*******************

For TOAST development, we prefer to keep our history as linear as possible.  Please avoid merging the upstream branch into your development branch.  Instead, use ``git fetch`` and ``git rebase`` (or ``git pull --rebase``) when updating branches from the upstream repo.  When merging large branches with many changes, please rebase those branches against current main before a pull request is reviewed / merged.  Small or trivial pull requests against recent versions of main are fine to merge without rebasing.
