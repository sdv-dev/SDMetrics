# Release workflow

This document describes the steps required to perform a full release of **SDMetrics**.

## Pre-requisites

Before starting to prepare an **SDMetrics** release make sure of the following points:

1. All the changes that need to be included in the current release have been added to
   the [sdv-dev/SDMetrics repository](https://github.com/sdv-dev/SDMetrics) `main`
   branch.
2. All the issues related to code changes which were closed after the latest release
   have been assigned to the current milestone, given a `bug` or `enhancement` tag
   and assigned to the developer or developers who contributed in resolving it.
3. All the Pull Requests that have been merged since the latest release are
   directly related to one of the issues assigned to the Milestone. If there is a
   Pull Requests for which no issue exist, you can treat the Pull Request itself as
   an issue and assign a developer, a tag and a milestone to it directly.
4. The Milestone that corresponds to the release does not have any issue that is still open.
   If there is any, either close it (if possible), or deassign it from the Milestone.
5. The latest build for the `main` branch performed by the CI systems (Github Actions
   and Travis) was successful. If the builds are more than a couple of days old, re-trigger
   them and wait for them to finish successfully.

## Clone the latest version of **SDMetrics** and test it

1. Clone the repository from scratch in a new folder:

```bash
git clone git@github.com:sdv-dev/SDMetrics SDMetrics.release
```

2. Create a fresh `virtualenv`, activate it and run `make install-develop` inside the repository
   folder (Note: the example uses plain `virtualenv`, but `virtualenvwrapper` or `conda` are
   valid alternatives:

```bash
cd SDMetrics.release
virtualenv venv
source venv/bin/activate
make install-develop
```

3. Test everything locally

```bash
make test-all
```

## (Optional) Bump version

If the release that you are about to make contained any API or relevant dependency changes,
you will need to bump the `minor` or `major` version of **SDMetrics**, and now would be the
right moment to do so.

For this, execute the `bumpversion-minor` or `bumpversion-major` make target:

```bash
make bumpversion-minor
```

## Make a release candidate

Before making the actual release, we will make a `release-candidate` which we will use to
test other libraries that depend on **SDMetrics**.

To do so, run:

```bash
make release-candidate
```

When asked to do so, type `confirm` and press enter, and also type your PyPI username and
password if requested.

If the process succeeds, do not forget to push back to GitHub afterwards:

```bash
git push
```

## Test the libraries that depend on **SDMetrics** using the release candidate

At this point you should test any known dependant libraries, like **SDV**, using the release
candidate that we just uploaded to PyPI.

For this, install the development version of SDV (or the library that you will test) inside a
new `virtualenv` and run its tests.

```bash
git clone git@github.com:sdv-dev/SDV SDV.SDMetrics
cd SDV.SDMetrics
virtualenv venv
source bin/activate
make install-develop
pip install --pre sdmetrics
make test-all
```

## Write the release notes inside `HISTORY.md`

If everything succeeded, you are ready to do the final release, which need to include its
corresponding release notes.

For this, go to the [SDMetrics Milestones page](https://github.com/sdv-dev/SDMetrics/milestones)
and click on the milestone that corresponds to the version that you are about to release to
get the list of all the Issues and Pull Requests that have been assigned to it, which you can
select and copy to use as a template for the release notes.

Now open the `HISTORY.md` file, create a new release section at the top with a second level
title which equals to `## <release-version> - <release-date>`, and then paste the issues list
that you just copied below and edit it to add:

* A short comprehensive summary of what's included on this release.
* A gratitude note to the developers that contributed to the project during this milestone,
  using their Github usernames.
* One or more third level titles to either introduce an `### Issues resolved` section or multiple
  sections, like `### Bugs Fixed`, `### New Features`, etc. Choose one option or the other
  depending on the amount of issues that were closed.
* Edit the issue titles that you pasted before, grouping them by types of issues, adding the
  links to the Github issues and adding a `by @username` at the end indicating who resolved it.

After your edits, the top of the file should look like this:

```
# History

## X.Y.Z - YYYY-MM-DD

<SHORT DESCRIPTION>

Thanks to @<CONTRIBUTOR-USER-NAMES...> for contributing to this release.

### New Features

* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/SDMetrics/issues/<issue>) by @resolver

### General Improvements

* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/SDMetrics/issues/<issue>) by @resolver

### Bug Fixed

* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/SDMetrics/issues/<issue>) by @resolver
```

In case of doubt, loo at the previous release notes and try to follow a similar style.

Once this is done, commit the change directly to `main` with the message `Add release notes for
vX.Y.Z`.

## Make the final release

Once everything else is ready, make the actual release with the command:

```bash
make release
```

Type `confirm` when asked, and if required enter your PyPI username and password.

## Add the release notes to Github

After the release is made, copy the text that you just added to HISTORY.md and go the [Releases](
https://github.com/sdv-dev/SDMetrics/releases) section in Github. You should find the release
that you just made at the top, without description. Click on it and edit it, and then paste
the HISTORY.md that you just copied on the description box and set
`v<release-version> - <release-date>` as the title (mind the proceeding `v`).

## Close the milestone and create the new one

At this point, go back to the [SDMetrics Milestones page](https://github.com/sdv-dev/SDMetrics/milestones)
and `close` the one that corresponds to the release that we just made.

Also, if they do not exist yet, create the next patch and minor milestones.
