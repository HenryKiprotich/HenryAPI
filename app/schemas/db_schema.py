
USERMASTER_SCHEMA = """
Table: "UserMaster"
Columns:
- "ID" (integer, primary key)
- "FirstName" (character varying)
- "OtherName" (character varying)
- "PhoneNo" (character varying(20), unique)
- "Email" (character varying(255), unique)
- "Password" (text)
- "UserTypeID" (integer)
- "Status" (character varying(50), e.g., '1', '0')
- "PayStatus" (boolean)
- "Latitude" (double precision)
- "Longitude" (double precision)
- "CountyID" (integer)
- "SubCountyID" (integer)
- "WardID" (integer)
- "LocationID" (integer)
- "CreatedAt" (timestamp without time zone)

Notes:
- Always use double quotes around table and column names in SQL queries, e.g. SELECT * FROM "UserMaster" WHERE "Status" = 1
"""