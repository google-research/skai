# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A script for granting / revoking admin access to EagleEye.

If --is_admin is present, grants admin access by setting the {'admin':
True} custom claim for the user; if not, unsets all custom claims.

Expects `GOOGLE_CLOUD_PROJECT` env variable to be set to the
appropriate GCP project name, and `GOOGLE_APPLICATION_CREDENTIALS` to
be set to the path to of a service account key json file with required
permission to access the GCP Admin SDK.

Example invocation:

python grant_admin.py --email=your@email.com
"""

from absl import app
from absl import flags
import firebase_admin
from firebase_admin import auth


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'email', None, 'Email of the user to grant / revoke access.', required=True
)
flags.DEFINE_boolean('is_admin', True, help='If present, grants admin access.')


def main(_):
  firebase_admin.initialize_app()

  user = auth.get_user_by_email(FLAGS.email)
  print(
      f'User (email): {FLAGS.email}, user id: {user.uid}, claims:'
      f' {user.custom_claims}'
  )
  claims = {'admin': True} if FLAGS.is_admin else {}
  auth.set_custom_user_claims(user.uid, claims)
  if FLAGS.is_admin:
    print(f'Granted admin access for user: {FLAGS.email} (uid: {user.uid})')
  else:
    print(f'Revoked admin access for user: {FLAGS.email} (uid: {user.uid})')


if __name__ == '__main__':
  app.run(main)
