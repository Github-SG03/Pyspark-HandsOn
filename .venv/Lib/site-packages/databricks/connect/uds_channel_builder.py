#
# DATABRICKS CONFIDENTIAL & PROPRIETARY
# __________________
#
# Copyright 2021-present Databricks, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property of Databricks, Inc.
# and its suppliers, if any.  The intellectual and technical concepts contained herein are
# proprietary to Databricks, Inc. and its suppliers and may be covered by U.S. and foreign Patents,
# patents in process, and are protected by trade secret and/or copyright law. Dissemination, use,
# or reproduction of this information is strictly forbidden unless prior written permission is
# obtained from Databricks, Inc.
#
# If you view or obtain a copy of this information and believe Databricks, Inc. may not have
# intended it to be made available, please promptly report it to Databricks Legal Department
# @ legal@databricks.com.
#

import grpc
from typing import Dict, Optional

from pyspark.sql.connect.client import ChannelBuilder


class UDSChannelBuilder(ChannelBuilder):
    def __init__(self, connectionString: str, params: Optional[Dict[str, str]] = None):
        super().__init__(params=params)
        if connectionString[:7] != "unix://":
            raise AttributeError("URL scheme must be set to `unix`.")
        self.path = connectionString.split(";")[0]

    def toChannel(self) -> grpc.Channel:
        return self._insecure_channel(self.path)

    @property
    def host(self) -> str:
        return self.path
