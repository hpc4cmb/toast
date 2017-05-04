
SHELL = /bin/bash

# Check required environment variables

ifndef CONFIG
  CONFIG := $(error CONFIG undefined)undefined
endif

gitdesc := $(shell git describe --tags --dirty --always | cut -d "-" -f 1)
gitcnt := $(shell git rev-list --count HEAD)

TOASTVERSION := "$(gitdesc).dev$(gitcnt)"
ifndef VERSION
  VERSION := "$(gitdesc).dev$(gitcnt)"
endif

# Make seems to delete these...

.PRECIOUS : $(wildcard conf/*)

# Override arcane Make suffix rule for SCCS which 
# wipes our config scripts...

% : %.sh

# Config related files

CONFIG_FILE := conf/$(CONFIG)

# The files with the build rules for each dependency

rules_full := $(wildcard rules/*)
rules_sh := $(foreach rl,$(rules_full),$(subst rules/,,$(rl)))
rules := $(foreach rl,$(rules_sh),$(subst .sh,,$(rl)))

# For depending on the helper scripts and templates

TOOLS := $(wildcard ./tools/*)

# Based on the config file name, are we building a docker file
# or an install script?

DOCKERCHECK := $(findstring docker,$(CONFIG))
ifeq "$(DOCKERCHECK)" "docker"
  SCRIPT := Dockerfile_$(CONFIG)
else
  SCRIPT := install_$(CONFIG).sh
  ifndef PREFIX
    PREFIX := $(error PREFIX undefined)undefined
  endif
endif

# Allow manually specifying the modulefiles directory.
# Otherwise install to PREFIX/modulefiles

ifndef MODULEDIR
  MODULEDIR := "$(PREFIX)/modulefiles"
endif


.PHONY : help script clean


help :
	@echo " "
	@echo " Before using this Makefile, set the CONFIG and PREFIX environment"
	@echo " variables.  The VERSION environment variable is optional.  The"
	@echo " following targets are supported:"
	@echo " "
	@echo "    script  : Build the appropriate install script or Dockerfile."
	@echo "    clean   : Clean all generated files."
	@echo " "


script : $(CONFIG_FILE) $(TOOLS) $(SCRIPT)
	@echo "" >/dev/null


Dockerfile_$(CONFIG) : $(CONFIG_FILE) $(TOOLS) Dockerfile.template
	@./tools/apply_conf.sh Dockerfile.template "Dockerfile_$(CONFIG)" "$(CONFIG_FILE)" "$(PREFIX)" "$(VERSION)" "$(MODULEDIR)" yes


install_$(CONFIG).sh : $(CONFIG_FILE) $(TOOLS) install.template
	@./tools/apply_conf.sh install.template "install_$(CONFIG).sh" "$(CONFIG_FILE)" "$(PREFIX)" "$(VERSION)" "$(MODULEDIR)" no \
	&& chmod +x "install_$(CONFIG).sh" \
	&& ./tools/gen_modulefile.sh tools/modulefile.in "install_$(CONFIG).sh.modtemplate" "$(CONFIG_FILE).module" \
	&& ./tools/apply_conf.sh "install_$(CONFIG).sh.modtemplate" "install_$(CONFIG).sh.module" "$(CONFIG_FILE)" "$(PREFIX)" "$(VERSION)" "$(MODULEDIR)" no \
	&& ./tools/apply_conf.sh tools/version.in "install_$(CONFIG).sh.modversion" "$(CONFIG_FILE)" "$(PREFIX)" "$(VERSION)" "$(MODULEDIR)" no


Dockerfile.template : tools/Dockerfile.in $(rules_full) $(TOOLS)
	@./tools/gen_template.sh tools/Dockerfile.in Dockerfile.template "$(rules)" RUN


install.template : tools/install.in $(rules_full) $(TOOLS)
	@./tools/gen_template.sh tools/install.in install.template "$(rules)"


clean :
	@rm -f Dockerfile_* Dockerfile.template install_* install.template

