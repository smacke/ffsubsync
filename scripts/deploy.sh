#!/usr/bin/env bash

# ref: https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

if ! git diff-index --quiet HEAD --; then
    echo "dirty working tree; please clean or commit changes"
    exit 1
fi

if ! git describe --exact-match --tags HEAD > /dev/null; then
    echo "current revision not tagged; please deploy from a tagged revision"
    exit 1
fi

current="$(python -c 'import versioneer; print(versioneer.get_version())')"
[[ $? -eq 1 ]] && exit 1

latest="$(git describe --tags $(git rev-list --tags --max-count=1))"
[[ $? -eq 1 ]] && exit 1

if [[ "$current" != "$latest" ]]; then
    echo "current revision is not the latest version; please deploy from latest version"
    exit 1
fi

expect <<EOF
set timeout -1

spawn twine upload dist/*

expect "Enter your username:"
send -- "$(lpass show 937494930560669633 --username)\r"

expect "Enter your password:"
send -- "$(lpass show 937494930560669633 --password)\r"
expect
EOF

branch="$(git branch --show-current)"
git checkout latest
git rebase "$branch"
git push -f
git checkout "$branch"

git push --tags
