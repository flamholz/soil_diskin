#!/bin/bash

# Build the documentation using FORD

FORD_MD='project.md'

no_search=""
quiet=""
verbose="false"
clean="false"
while [ $# -gt 0 ]; do
  case $1 in
    -f)
      no_search="--no-search"
      shift
      ;;
    -q)
      quiet="-q"
      shift
      ;;
    -v)
      verbose="true"
      shift
      ;;
    -c)
      clean="true"
      shift
      ;;
  esac
done

if [[ $clean == true ]]; then
  rm -rf 3rdparty
  rm -rf doc/html
  echo "Removed 3rdparty and doc/html directories"
  exit
fi

mkdir 3rdparty >& /dev/null || true

if ! hash python3 2>/dev/null; then
  echo "python3 not found"
  exit 1
fi

PYTHON_VERSION=3.10
export UV_TOOL_DIR=`pwd`/3rdparty/uv
export UV_TOOL_BIN_DIR=${UV_TOOL_DIR}/bin
export UV=${UV_TOOL_BIN_DIR}/uv
export UV_LINK_MODE=symlink
if [[ ! -f $UV ]]; then
  if [[ $verbose == true ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=${UV_TOOL_BIN_DIR} INSTALLER_NO_MODIFY_PATH=1 sh 
  else
    ( curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=${UV_TOOL_BIN_DIR} INSTALLER_NO_MODIFY_PATH=1 sh ) &>/dev/null
  fi
fi
if [[ ! -f $UV ]]; then
  echo "uv not found"
  exit 1
fi

#${UV} python install 3.10

#${UV} tool install --with lxml ford@v6.2.5 # latest v6
#${UV} tool install --with lxml ford        # new v7
if [[ $verbose == true ]]; then
  #${UV} tool install --python $PYTHON_VERSION --with lxml --from https://github.com/RMShur/ford.git ford
  ${UV} tool install --with lxml --from https://github.com/RMShur/ford.git ford
else
  #${UV} tool install --python $PYTHON_VERSION --with lxml --from https://github.com/RMShur/ford.git ford 2>/dev/null
  ${UV} tool install --with lxml --from https://github.com/RMShur/ford.git ford 2>/dev/null
fi
FORD=${UV_TOOL_BIN_DIR}/ford

if grep 'graph: *true' ${FORD_MD}; then
  if ! hash dot; then
    echo "graph: true set in ${FORD_MD} but no installation of graphviz found"
    exit 1
  fi
fi

output_dir=$(grep output_dir project.md | cut -d' ' -f2)

rev=`git rev-parse --short HEAD`
#rev=`git log --pretty='format:%h %D' --first-parent | grep HEAD | cut -f1 -d' '`
[[ -z $(git status -suno) ]] && dirty="" || dirty="(dirty)"
if [[ -z $GITLAB_CI ]]; then
  branch=`git log --pretty='format:%h %D' --first-parent | grep HEAD | cut -f4 -d' ' | sed 's/,//'`
else
  branch=$CI_COMMIT_REF_NAME
fi

echo "Building documentation (Revision:${rev}${dirty} Branch:${branch}) ..."
mkdir pp_temp$$
python3 scripts/dsl4jsb/dsl4jsb.py -n -d src -t pp_temp$$ -k
find pp_temp$$ -type f -name '*.f90' -exec \
    perl -i -pe 's/!> /!> Summary: / if $. == 1;' \
            -pe 's/icon-model.org/<https:\/\/icon-model.org>  / if $. <= 20;' \
            -pe 's/!> ICON-Land/!>#### ICON-Land/ if $. <= 20;' \
            -pe 's/MPI-BGC$/MPI-BGC  / if $. <= 20;' \
            -pe 's/AUTHORS.md$/AUTHORS.md  / if $. <= 20;' \
            -pe 's/license information$/license information  / if $. <= 20;' \
    '{}' \;
$FORD $no_search $quiet -r "Revision:${rev}${dirty} Branch:${branch}" -d pp_temp$$ ${FORD_MD}
find $output_dir -type f -name '*.html' -exec perl -pi -e 's/was developed by/is developed by/;' '{}' \;
rm -rf pp_temp$$

