#!/usr/bin/env bash

treepid() {
    local pid sep
    sep=${sep:-'-'}
    pid=${pid:-$1}

    ps -eo ppid,pid --no-headers | awk -vroot=$pid -vsep="$sep" '
        function dfs(u) {
            if (pids)
                pids = pids sep u;
            else
                pids = u;
            if (u in edges)
                for (v in edges[u])
                    dfs(v);
        }
        {
            edges[$1][$2] = 1;
            if ($2 == root)
                root_isalive = 1;
        }
        END {
            if (root_isalive)
                dfs(root);
            if (pids)
                print pids;
        }'
}

treepid "$@" | awk -F "-" '{print $NF}' | xargs kill -9